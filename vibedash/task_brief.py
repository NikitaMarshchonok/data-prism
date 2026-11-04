"""User-confirmed task context; no LLM inference or dataset approval."""
import hashlib
import json
import re
from collections.abc import Mapping

from .task_capabilities import assess_task_request


CONTRACT = 'confirmed-task-brief-v1'
MAX_REVIEW_BYTES = 16 * 1024
FIELDS = {'task_type', 'question', 'observation_unit', 'deliverable', 'confirmed'}
FORM_FIELDS = ('task_observation_unit', 'task_deliverable', 'task_brief_confirmed')
DELIVERABLES = {
    'descriptive_analysis': ('evidence_summary', 'A descriptive evidence summary with calculations and limitations.'),
    'daily_forecast': ('forecast_evaluation', 'A daily forecast evaluation against simple baselines, plus an exploratory future forecast.'),
}


class TaskBriefError(ValueError):
    def __init__(self, assessment):
        self.assessment = assessment
        super().__init__(assessment['message'])


def _error(message):
    return TaskBriefError({'contract': CONTRACT, 'status': 'invalid_request',
                           'ready_for_execution': False, 'message': message, 'questions': []})


def _text(value, maximum, label):
    if not isinstance(value, str) or len(value) > maximum:
        raise _error(f'{label} must be text of at most {maximum} characters.')
    try:
        value.encode('utf-8')
    except UnicodeError as error:
        raise _error(f'{label} must contain valid text.') from error
    return value.strip()


def review_task_brief(request):
    """Ask only for missing explicit context, without pretending to infer intent."""
    if not isinstance(request, Mapping) or set(request) - FIELDS:
        raise _error('Provide only the supported task brief fields.')
    task_type = request.get('task_type')
    admission = assess_task_request(task_type, workflow='analysis_jobs')
    result = {'contract': CONTRACT, 'status': 'needs_clarification', 'ready_for_execution': False,
              'message': 'Review the task and answer the missing questions.', 'questions': []}
    if admission['status'] != 'requires_dataset_review':
        return result | {'status': admission['status'], 'message': admission['message'],
                         'questions': [{'field': 'task_type', 'question': question}
                                       for question in admission['questions']]}
    question = _text(request.get('question', ''), 4000, 'Analysis question')
    unit = _text(request.get('observation_unit', ''), 240, 'Observation unit')
    deliverable = _text(request.get('deliverable', ''), 64, 'Requested result')
    confirmed = request.get('confirmed', False)
    if type(confirmed) is not bool:
        raise _error('Task confirmation must be a boolean.')
    expected, description = DELIVERABLES[task_type]
    for missing, field, text in (
        (not question, 'question', 'What question should the work answer or which decision should it support?'),
        (not unit, 'observation_unit', 'What does one row represent, for example one day for one location or one support request?'),
        (deliverable != expected, 'deliverable', f'Confirm the supported result: {description}'),
        (not confirmed, 'confirmed', 'Review and confirm the question, row meaning, expected result and column meanings.'),
    ):
        if missing:
            result['questions'].append({'field': field, 'question': text})
    result.update(task_type=task_type, expected_deliverable=expected, proposed_result=description,
                  limitations=admission['limitations'],
                  interpretation_source='explicit_user_context', business_understanding='not_independently_verified')
    if not result['questions']:
        result.update(status='ready_for_dataset_review',
                      message='The task brief is complete and user-confirmed; data and methodological checks still apply.')
    return result


def parse_review_json(raw):
    if not isinstance(raw, bytes) or len(raw) > MAX_REVIEW_BYTES:
        raise _error('The task brief review exceeds its byte limit.')
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate field')
            result[key] = value
        return result
    def constant(_value):
        raise ValueError('non-finite JSON')
    try:
        return json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise _error('Provide a bounded JSON object without duplicate fields or non-finite values.') from error


def prepare_task_brief(form, *, task_type, question):
    """Opt-in form boundary; mode-only legacy clients keep their current path."""
    if not any(field in form for field in FORM_FIELDS):
        return None
    draft = {'task_type': task_type, 'question': question,
             'observation_unit': form.get('task_observation_unit', ''),
             'deliverable': form.get('task_deliverable', ''),
             'confirmed': form.get('task_brief_confirmed') == 'yes'}
    review = review_task_brief(draft)
    if review['status'] != 'ready_for_dataset_review':
        raise TaskBriefError(review)
    return {**draft, 'question': draft['question'].strip(), 'observation_unit': draft['observation_unit'].strip(),
            'deliverable': draft['deliverable'].strip()}


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def bind_task_brief(draft, semantics_sha256):
    if review_task_brief(draft)['status'] != 'ready_for_dataset_review':
        raise _error('A complete confirmed task brief is required.')
    if not isinstance(semantics_sha256, str) or not re.fullmatch('[0-9a-f]{64}', semantics_sha256):
        raise _error('Bind the brief to a server-validated semantic contract.')
    result = {'contract': CONTRACT, **draft, 'semantics_sha256': semantics_sha256}
    result['sha256'] = _fingerprint(result)
    return result


def validate_stored_task_brief(value, *, task_type, question, semantics_sha256):
    if not isinstance(value, Mapping) or value.get('contract') != CONTRACT:
        raise _error('The stored task brief is missing or unsupported.')
    draft = {field: value.get(field) for field in FIELDS}
    draft.update(task_type=task_type, question=question)
    expected = bind_task_brief(draft, semantics_sha256)
    if dict(value) != expected:
        raise _error('The task brief no longer matches the confirmed question, task or column meanings.')
    return expected


def task_brief_audit_view(brief):
    """No user question or row-description text in the aggregate audit record."""
    return {key: brief[key] for key in ('contract', 'sha256', 'task_type', 'deliverable', 'confirmed', 'semantics_sha256')}
