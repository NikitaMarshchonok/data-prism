"""Industry-independent task admission for confirmed VibeDash workflows.

This is an explicit capability boundary, not a free-text intent classifier or
proof that a particular dataset/task is suitable. Admission precedes existing
dataset, semantic, availability and decision-policy checks; it never trains.
"""
from copy import deepcopy
from collections.abc import Mapping


CONTRACT = 'vibedash-task-intake-v1'
WORKFLOWS = {'analysis_jobs', 'period_comparison_jobs'}

_CAPABILITIES = {
    'descriptive_analysis': {
        'status': 'available', 'workflow': 'analysis_jobs', 'analysis_mode': 'evidence',
        'requirements': ['approved_tabular_data', 'analysis_question', 'reviewed_column_meanings'],
        'limitations': ['Observational findings do not establish causation.',
                        'Anomaly review candidates and exploratory segments are not confirmed incidents or operational segments.'],
    },
    'daily_forecast': {
        'status': 'available', 'workflow': 'analysis_jobs', 'analysis_mode': 'forecast',
        'requirements': ['one_complete_daily_series', 'confirmed_numeric_target', 'confirmed_daily_dates',
                         'explicit_horizon', 'confirmed_fixed_target_reporting_delay', 'reviewed_decision_criteria'],
        'limitations': ['Only past target lags are used; external predictors, multiple entities, variable reporting delays and revised target vintages are unsupported.',
                        'No calibrated prediction interval, saved forecast model or automatic outcome monitoring is provided.'],
    },
    'period_comparison': {
        'status': 'available', 'workflow': 'period_comparison_jobs', 'analysis_mode': None,
        'requirements': ['two_approved_csv_snapshots', 'same_observation_unit', 'consistent_metric_definitions',
                         'reviewed_period_semantics'],
        'limitations': ['Comparisons are observational, not estimates of causal effects.'],
    },
    'classification': {'status': 'not_available', 'workflow': None, 'analysis_mode': None,
                       'requirements': [], 'limitations': ['A confirmed standalone VibeDash classification workflow is not available.']},
    'regression': {'status': 'not_available', 'workflow': None, 'analysis_mode': None,
                   'requirements': [], 'limitations': ['A confirmed standalone VibeDash independent-row regression workflow is not available.']},
    'segmentation': {'status': 'embedded_exploration', 'workflow': None, 'analysis_mode': None,
                     'requirements': [], 'limitations': ['Exploratory segmentation exists within descriptive analysis; no standalone validated segmentation task is offered.']},
    'anomaly_detection': {'status': 'embedded_exploration', 'workflow': None, 'analysis_mode': None,
                          'requirements': [], 'limitations': ['Descriptive analysis ranks review candidates; it does not confirm fraud, errors or incidents.']},
    'causal_analysis': {'status': 'not_available', 'workflow': None, 'analysis_mode': None,
                        'requirements': [], 'limitations': ['No confirmed causal-effect or experiment-design workflow is available.']},
}


def capability_catalog():
    return {'contract': CONTRACT, 'scope': 'confirmed_vibedash_workflows',
            'industry_independent': True, 'automatic_business_understanding': False,
            'professional_equivalence': 'not_established',
            'tasks': [{'task_type': name, **deepcopy(value)} for name, value in _CAPABILITIES.items()]}


def assess_task_request(task_type, *, workflow=None):
    """Return static clarification/admission guidance; do not inspect source rows."""
    result = {'contract': CONTRACT, 'task_type': None, 'workflow': None,
              'ready_for_execution': False, 'requirements': [], 'questions': []}
    if workflow is not None and (not isinstance(workflow, str) or workflow not in WORKFLOWS):
        return result | {'status': 'unsupported_workflow', 'message': 'Select a supported workflow.'}
    if task_type is None or (isinstance(task_type, str) and not task_type.strip()):
        return result | {'status': 'needs_clarification',
                         'message': 'Confirm the requested task before choosing an execution workflow.',
                         'questions': ['What result or decision do you need?',
                                       'Do you need a description, a future forecast, or a comparison of two periods?']}
    if not isinstance(task_type, str) or task_type not in _CAPABILITIES:
        return result | {'status': 'not_supported',
                         'message': 'The requested task is not in the supported capability catalogue. No substitute was selected.'}
    capability = deepcopy(_CAPABILITIES[task_type])
    result.update(task_type=task_type, workflow=capability['workflow'], requirements=capability['requirements'],
                  limitations=capability['limitations'])
    if capability['status'] != 'available':
        return result | {'status': 'not_supported',
                         'message': 'This standalone task is unavailable. Exploratory findings are not a substitute for it.'}
    if workflow is not None and workflow != capability['workflow']:
        return result | {'status': 'workflow_mismatch',
                         'message': 'This task requires its own supported workflow. No alternative task was selected.'}
    return result | {'status': 'requires_dataset_review',
                     'message': 'The workflow exists; its data, meanings, assumptions and decision criteria still require validation.'}


class TaskAdmissionError(ValueError):
    def __init__(self, assessment):
        self.assessment = assessment
        super().__init__(assessment['message'])


def resolve_analysis_task(form):
    """Use explicit task_type when present, preserving legacy mode-only clients."""
    explicit = 'task_type' in form
    if explicit:
        assessment = assess_task_request(form.get('task_type'), workflow='analysis_jobs')
        if assessment['status'] != 'requires_dataset_review':
            raise TaskAdmissionError(assessment)
        mode = _CAPABILITIES[assessment['task_type']]['analysis_mode']
        if 'analysis_mode' in form and form.get('analysis_mode') != mode:
            raise TaskAdmissionError(assessment | {'status': 'conflicting_task',
                                     'message': 'The selected task and analysis mode disagree. Confirm one task; no substitute was selected.'})
    else:
        mode = form.get('analysis_mode', 'evidence')
        if not isinstance(mode, str) or mode not in {'evidence', 'forecast'}:
            raise TaskAdmissionError({'contract': CONTRACT, 'status': 'not_supported',
                                     'ready_for_execution': False, 'message': 'Unsupported analysis mode.'})
        assessment = assess_task_request('daily_forecast' if mode == 'forecast' else 'descriptive_analysis',
                                         workflow='analysis_jobs')
    routing = {'contract': CONTRACT, 'task_type': assessment['task_type'], 'workflow': 'analysis_jobs',
               'intent_source': 'explicit_task_type' if explicit else 'legacy_analysis_mode'}
    return mode, routing


def validate_stored_task_routing(payload):
    """Recheck new-job routing in the worker; legacy jobs have no such record."""
    routing = payload.get('task_routing')
    if routing is None:
        return
    if not isinstance(routing, Mapping):
        raise ValueError('The stored task routing is invalid.')
    expected_type = 'daily_forecast' if payload.get('analysis_kind') == 'daily_forecast' else 'descriptive_analysis'
    expected = {'contract': CONTRACT, 'task_type': expected_type, 'workflow': 'analysis_jobs',
                'intent_source': routing.get('intent_source')}
    if (dict(routing) != expected or routing.get('intent_source') not in ('explicit_task_type', 'legacy_analysis_mode')
            or payload.get('analysis_kind') not in (None, 'daily_forecast')
            or (expected_type == 'daily_forecast' and not isinstance(payload.get('forecast_task'), Mapping))
            or (expected_type == 'descriptive_analysis' and payload.get('forecast_task') is not None)):
        raise ValueError('The stored task routing does not match the execution workflow.')
    if expected_type == 'daily_forecast' and routing['intent_source'] == 'explicit_task_type':
        task = payload['forecast_task']
        if not isinstance(task.get('decision_policy'), Mapping) or task.get('decision_policy_confirmed') is not True:
            raise ValueError('An explicit forecast task requires its confirmed decision policy.')
