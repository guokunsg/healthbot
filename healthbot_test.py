from jedi.settings import auto_import_modules
from langchain_core.messages import AIMessage
from healthbot import create_health_bot_graph
from unittest import mock
from unittest.mock import patch, Mock

import agent_query_topic


AI_MESSAGE_CLARIFICATION = AIMessage(
    content='',
    additional_kwargs={'tool_calls': [{'id': 'call_PSD7YZmyHqwVy36y52DWPph9', 'function': {'arguments': '{"message":"Could you please clarify what health topic or medical condition you’d like to learn about?"}', 'name': 'tool_clarification'}, 'type': 'function'}], 'refusal': None},
    response_metadata={'token_usage': {'completion_tokens': 32, 'prompt_tokens': 730, 'total_tokens': 762, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_8bda4d3a2c', 'id': 'chatcmpl-CD3qmMCovc7rZ5YJkz8GbE94mwwZD', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None},
    id='run--df1d1286-8aa3-46b3-8da7-08bbe7ff828b-0',
    tool_calls=[{'name': 'tool_clarification', 'args': {'message': 'Could you please clarify what health topic or medical condition you’d like to learn about?'}, 'id': 'call_PSD7YZmyHqwVy36y52DWPph9', 'type': 'tool_call'}],
    usage_metadata={'input_tokens': 730, 'output_tokens': 32, 'total_tokens': 762, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
)
AI_MESSAGE_TOOL_SEARCH = AIMessage(
    content='',
    additional_kwargs={'tool_calls': [{'id': 'call_GUdy43r4wkx9IZnORFkh9b2c', 'function': {'arguments': '{"keywords":"COVID-19 overview symptoms treatment prevention"}', 'name': 'tool_search_query'}, 'type': 'function'}], 'refusal': None},
    response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 773, 'total_tokens': 794, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_8bda4d3a2c', 'id': 'chatcmpl-CD3qo3SFxpBHWPeTAClS1VGaiZmVn', 'service_tier': 'default', 'finish_reason': 'tool_calls', 'logprobs': None},
    id='run--ae45aa07-85d4-422f-9e58-dc59fb0d4c42-0',
    tool_calls=[{'name': 'tool_search_query', 'args': {'keywords': 'COVID-19 overview symptoms treatment prevention'}, 'id': 'call_GUdy43r4wkx9IZnORFkh9b2c', 'type': 'tool_call'}],
    usage_metadata={'input_tokens': 773, 'output_tokens': 21, 'total_tokens': 794, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
)
AI_MESSAGE_SUMMARY = AIMessage(
    content='COVID-19, caused by the SARS-CoV-2 virus, is a respiratory illness that emerged in December 2019. It can range from mild to severe, and some individuals may not show any symptoms at all. Here’s a summary of key information about COVID-19:\n\n### What is COVID-19?\nCOVID-19 is a disease that primarily affects the respiratory system. It can lead to a variety of symptoms and complications, and it spreads easily from person to person.\n\n### Causes and Risk Factors\nCOVID-19 spreads through respiratory droplets when an infected person coughs, sneezes, or talks. People with underlying health conditions, older adults, and those who are unvaccinated are at a higher risk for severe illness.\n\n### Common Symptoms\nSymptoms of COVID-19 can vary widely but may include:\n- Fever or chills\n- Cough\n- Shortness of breath or difficulty breathing\n- Fatigue\n- Muscle or body aches\n- Loss of taste or smell\n- Sore throat\n- Congestion or runny nose\n- Nausea or vomiting\n- Diarrhea\n\n### Long COVID\nSome individuals may experience lingering symptoms for weeks or months after the initial infection, a condition known as long COVID. Symptoms can differ from those of the initial illness.\n\n### Treatment and Management\nWhile there is no specific antiviral treatment for COVID-19, supportive care is essential. This may include:\n- Rest and hydration\n- Over-the-counter medications for symptom relief\n- Hospitalization for severe cases, which may require oxygen therapy or mechanical ventilation\n\n### Prevention\nPreventive measures include:\n- Getting vaccinated\n- Wearing masks in crowded or indoor settings\n- Practicing good hand hygiene\n- Maintaining physical distance from others\n\nFor more detailed information, you can visit the [CDC COVID-19 page](https://www.cdc.gov/covid/index.html) or the [World Health Organization (WHO) COVID-19 page](https://www.who.int/health-topics/coronavirus).\n\nIf you have any more questions or need further information, feel free to ask!',
    additional_kwargs={'refusal': None},
    response_metadata={'token_usage': {'completion_tokens': 424, 'prompt_tokens': 3415, 'total_tokens': 3839, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': 'fp_8bda4d3a2c', 'id': 'chatcmpl-CD3qs0BKqwOnvEZtcpVR9oEBtkiGi', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None},
    id='run--47b19c39-0859-46f0-b1a7-c47f4cf4844c-0',
    usage_metadata={'input_tokens': 3415, 'output_tokens': 424, 'total_tokens': 3839, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
)
MOCK_CLARIFICATION_RESPONSE = "Covid"
MOCK_SEARCH_RESPONSE = {}

def setup_mocks(mocker, ai_messages):
    # Mock llm responses
    mock_llm = Mock()
    mocker.patch("agent_query_topic.create_llm", return_value=mock_llm)
    mock_llm_invoke = mocker.patch.object(mock_llm, 'invoke')
    # Return the ai messages in turn
    mock_llm_invoke.side_effect = lambda *args, **kwargs: ai_messages[mock_llm_invoke.call_count - 1]

    # Mock tool function calls
    mock_tool_clarification = mocker.patch("agent_query_topic.tool_clarification",
                                           wraps=agent_query_topic.tool_clarification)
    mock_tool_clarification.return_value = MOCK_CLARIFICATION_RESPONSE
    mock_tool_search = mocker.patch("agent_query_topic.tool_search_query",
                                    wraps=agent_query_topic.tool_search_query)
    mock_tool_search.return_value = MOCK_SEARCH_RESPONSE

    return mock_llm_invoke, mock_tool_clarification, mock_tool_search

def test_no_clarification_flow(mocker):
    init_topic = "Covid"
    ai_messages = [ AI_MESSAGE_TOOL_SEARCH, AI_MESSAGE_SUMMARY ]
    mock_llm_invoke, mock_tool_clarification, mock_tool_search = setup_mocks(mocker, ai_messages)

    # Create the graph and invoke the flow
    graph = create_health_bot_graph()
    config = {"configurable": {"thread_id": "1"}}
    final_state = graph.invoke({"user_query": init_topic}, config)

    assert final_state["search_summary"] == AI_MESSAGE_SUMMARY.content
    assert mock_llm_invoke.call_count == 2
    assert mock_tool_clarification.call_count == 0
    assert mock_tool_search.call_count == 1


def test_clarification_required(mocker):
    init_topic = "weather"
    ai_messages = [ AI_MESSAGE_CLARIFICATION, AI_MESSAGE_TOOL_SEARCH, AI_MESSAGE_SUMMARY ]

    mock_llm_invoke, mock_tool_clarification, mock_tool_search = setup_mocks(mocker, ai_messages)

    # Create the graph and invoke the flow
    graph = create_health_bot_graph()
    config = {"configurable": {"thread_id": "1"}}
    final_state = graph.invoke({"user_query": init_topic}, config)

    assert final_state["search_summary"] == AI_MESSAGE_SUMMARY.content
    assert mock_llm_invoke.call_count == 3
    assert mock_tool_clarification.call_count == 1
    assert mock_tool_search.call_count == 1