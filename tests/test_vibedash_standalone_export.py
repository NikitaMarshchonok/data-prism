"""An exported report must render from its embedded assets, not the server."""
from html.parser import HTMLParser
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from flask import Flask, render_template
import pandas as pd
from plotly.offline import get_plotlyjs

from vibedash.exporter import make_single_file_html, _get_plotly_js
from vibedash.generator_bridge import generate_dashboard_data
from vibedash.spec import Chart, Metric, VizSpec


ROOT = Path(__file__).resolve().parents[1]


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.declarations = []
        self.feed(html)
    def handle_decl(self, value): self.declarations.append(value)
    def handle_starttag(self, tag, attrs): self.tags.append((tag, dict(attrs)))


class StandaloneExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=str(ROOT / 'static'))
        for endpoint in ('index', 'analysis_history', 'decision_cases', 'export'):
            cls.app.add_url_rule('/' + endpoint, endpoint='vibedash.' + endpoint, view_func=lambda: '')
        cls.frame = pd.DataFrame({'season': [1, 2, 3, 4], 'cnt': [10, 20, 30, 40],
                                  'dteday': pd.date_range('2020-01-01', periods=4)})
        cls.spec = VizSpec(title='Counts <script>untrusted</script>',
                           metrics=[Metric(title='Total count', expr='sum(cnt)', fmt='number')],
                           charts=[Chart(type='bar', group='season', y='cnt', agg='sum'),
                                   Chart(type='hist', x='cnt'), Chart(type='line', x='dteday', y='cnt')])
        cls.dashboard = generate_dashboard_data(cls.frame, cls.spec, prompt='Analyze cnt over dteday')

    def render(self, export=True):
        with self.app.test_request_context('/'):
            return render_template('vibedash_evidence.html', dashboard_data=self.dashboard,
                                   viz_spec=self.spec, filename='test.csv', prompt='Analyze cnt over dteday',
                                   session_id='test', export_mode=export,
                                   audit_manifest={'analysis_contract': 'vibedash-evidence-v3',
                                                   'engine_version': 'test-version',
                                                   'dataset': {'analyzed_rows': 4, 'content_sha256': 'a' * 64,
                                                               'schema_sha256': 'b' * 64},
                                                   'specification': {'sha256': 'c' * 64}})

    def package(self, html=None):
        return make_single_file_html(self.render() if html is None else html,
                                     css_paths=[ROOT / 'static/vibedash_preview.css'],
                                     js_paths=[ROOT / 'static/vibedash_charts.js'])

    def test_exactly_one_document_without_network_or_relative_asset_tags(self):
        html = self.package()
        self.assertIn('<meta charset="UTF-8">', html[:1024])
        doc = Document(html)
        for tag in ('html', 'head', 'body'):
            self.assertEqual(sum(name == tag for name, _ in doc.tags), 1)
        self.assertEqual(len(doc.declarations), 1)
        self.assertFalse(any(name == 'script' and attrs.get('src') for name, attrs in doc.tags))
        self.assertFalse(any(name == 'link' and attrs.get('href') for name, attrs in doc.tags))
        self.assertFalse(any(name in {'form', 'iframe', 'object', 'embed'} for name, _ in doc.tags))
        self.assertNotIn('/static/vibedash_charts.js', html)
        self.assertNotIn('/static/vibedash_preview.css', html)

    def test_runtime_matches_installed_serializer_and_is_embedded_once_before_figures(self):
        html = self.package()
        self.assertEqual(html.count('data-data-prism-runtime="plotly"'), 1)
        self.assertIn(get_plotlyjs().splitlines()[1], html)
        self.assertLess(html.index('data-data-prism-runtime="plotly"'), html.index('Plotly.newPlot('))
        self.assertNotIn('<script src="https://cdn.plot.ly', html)

    def test_kpi_figures_and_fingerprints_are_preserved_verbatim(self):
        html = self.package()
        self.assertIn('100', html)
        for chart in self.dashboard['charts']:
            self.assertIn(chart['html'], html)
        self.assertIn('"type":"category"', html)
        self.assertIn('Statistical inference withheld', html)
        self.assertIn('test-version', html)
        for fingerprint in ('a' * 64, 'b' * 64, 'c' * 64): self.assertIn(fingerprint, html)

    def test_offline_document_has_network_denial_and_inlined_resize_controller(self):
        doc = Document(self.package())
        policy = next(attrs['content'] for tag, attrs in doc.tags
                      if tag == 'meta' and attrs.get('http-equiv') == 'Content-Security-Policy')
        self.assertIn("connect-src 'none'", policy)
        self.assertIn("default-src 'none'", policy)
        self.assertNotIn('unsafe-eval', policy)
        self.assertIn((ROOT / 'static/vibedash_charts.js').read_text().strip(), self.package())

    def test_live_page_keeps_assets_and_controls_export_page_has_no_dead_route_links(self):
        live, exported = self.render(False), self.package()
        self.assertIn('src="/static/vibedash_charts.js"', live)
        self.assertIn('Export HTML', live)
        self.assertNotIn('Export HTML', exported)
        self.assertFalse(any(tag == 'a' and attrs.get('href', '').startswith('/')
                             for tag, attrs in Document(exported).tags))

    def test_untrusted_labels_remain_escaped(self):
        html = self.package()
        self.assertIn('&lt;script&gt;untrusted&lt;/script&gt;', html)
        self.assertNotIn('<script>untrusted</script>', html)

    def test_missing_local_assets_fail_instead_of_partial_success(self):
        for kind in ('css_paths', 'js_paths'):
            with self.subTest(kind=kind), self.assertRaises(FileNotFoundError):
                make_single_file_html(self.render(), **{kind: [ROOT / 'static/DOES-NOT-EXIST.asset']})

    def test_relative_and_remote_resources_are_rejected_without_fetching(self):
        for resource in ('<script src="https://example.invalid/a.js"></script>',
                         '<script src="/static/a.js"></script>', '<link rel="stylesheet" href="x.css">',
                         '<img src="https://example.invalid/a.png">', '<iframe src="x"></iframe>'):
            with self.subTest(resource=resource), self.assertRaises(ValueError):
                make_single_file_html('<!doctype html><html><head>' + resource + '</head><body></body></html>')

    def test_nested_and_incomplete_documents_are_rejected(self):
        for html in ('<!doctype html><html><head></head><body><html></html></body></html>',
                     '<!doctype html><html><body>Missing head</body></html>',
                     '<!doctype html><head></head>'):
            with self.subTest(html=html), self.assertRaises(ValueError): make_single_file_html(html)

    def test_fragment_support_does_not_wrap_a_complete_document_twice(self):
        html = make_single_file_html('<main>Report fragment</main>')
        self.assertEqual(sum(tag == 'html' for tag, _ in Document(html).tags), 1)
        self.assertIn('<main>Report fragment</main>', html)

    def test_asset_closing_sequences_cannot_create_extra_elements(self):
        with TemporaryDirectory() as directory:
            css, js = Path(directory) / 'test.css', Path(directory) / 'test.js'
            css.write_text('/* </style><script>bad()</script> */')
            js.write_text('const label = "</script><script>bad()</script>";')
            html = make_single_file_html(self.render(), css_paths=[css], js_paths=[js])
        self.assertIn('<\\/style', html)
        self.assertIn('<\\/script', html)
        # Text inside a CSS comment is not an HTML element. Count parsed
        # scripts, not substrings in trusted raw-text assets.
        baseline_scripts = sum(tag == 'script' for tag, _ in Document(self.package()).tags)
        actual_scripts = sum(tag == 'script' for tag, _ in Document(html).tags)
        self.assertEqual(actual_scripts, baseline_scripts)

    def test_offline_packaging_is_independent_of_working_directory(self):
        with TemporaryDirectory() as directory, patch('os.getcwd', return_value=directory):
            self.assertIn('data-data-prism-runtime="plotly"', self.package())

    def test_plotly_bundle_is_cached_without_changing_packaged_output(self):
        _get_plotly_js.cache_clear()
        try:
            with patch('vibedash.exporter.get_plotlyjs', return_value='/* trusted-runtime */') as bundle:
                first = _get_plotly_js()
                second = _get_plotly_js()
                self.assertEqual(first, second)
                self.assertEqual(bundle.call_count, 1)
        finally:
            _get_plotly_js.cache_clear()


if __name__ == '__main__': unittest.main()
