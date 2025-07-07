async def forward_22(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction2 = "Subtask 2: Define problem variables: list length n, list elements x_i, and restate constraints (sum=30, mode=9, median integer not in list)"
    cot_sc_desc = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc,
        n_repeat=self.max_sc
    )
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, defining problem variables, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction1 = "Subtask 1: Derive the median position and possible median value given n and the constraints"
    critic_instruction1 = "Please review the derivation of median position and possible median values for correctness and completeness."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo] + results2['list_thinking'] + results2['list_answer'],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, deriving median position and values, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction0 = "Subtask 0: Optionally quantify the number of feasible list-length configurations satisfying the sum and preliminary median constraints"
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0 = await self.cot(
        subtask_id="subtask_0",
        cot_agent_desc=cot_agent_desc0
    )
    sub_tasks.append(f"Subtask 0 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    agents.append(f"CoT agent {results0['cot_agent'].id}, quantifying feasible list-length configurations, thinking: {results0['thinking'].content}; answer: {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    candidate_lists = []
    for i in range(1):
        cot_instruction3 = "Subtask 3: Generate candidate lists summing to 30 with unique mode 9 and median not present in the list"
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results3 = await self.cot(
            subtask_id="subtask_3",
            cot_agent_desc=cot_agent_desc3
        )
        sc_cot_instruction3 = "Subtask 3: Refine candidate lists generation with self-consistency to ensure constraints are met"
        sc_cot_desc3 = {
            'instruction': sc_cot_instruction3,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results3_sc = await self.sc_cot(
            subtask_id="subtask_3_sc",
            cot_sc_desc=sc_cot_desc3,
            n_repeat=self.max_sc
        )
        answer_generate_instruction3 = "Subtask 3: Generate final candidate lists from refined outputs"
        answer_generate_desc3 = {
            'instruction': answer_generate_instruction3,
            'input': [taskInfo, results3_sc['thinking'], results3_sc['answer']],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results3_final = await self.answer_generate(
            subtask_id="subtask_3_final",
            cot_agent_desc=answer_generate_desc3
        )
        candidate_lists.append(results3_final['answer'].content)
        sub_tasks.append(f"Subtask 3 output: thinking - {results3_final['thinking'].content}; answer - {results3_final['answer'].content}")
        agents.append(f"AnswerGenerate agent {results3_final['cot_agent'].id}, generating candidate lists, thinking: {results3_final['thinking'].content}; answer: {results3_final['answer'].content}")
        logs.append(results3_final['subtask_desc'])

    aggregate_instruction4 = "Subtask 4: Consolidate multiple generated candidate lists into a unified set of viable solutions"
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo] + candidate_lists,
        'temperature': 0.0,
        'context': ["user query", "candidate lists"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating candidate lists, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    programmer_instruction5 = "Subtask 5: Validate each candidate list against the sum, unique mode, and median constraints"
    programmer_desc5 = {
        'instruction': programmer_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "consolidated candidate lists"],
        'entry_point': "validate_candidates"
    }
    results5 = await self.programmer(
        subtask_id="subtask_5",
        programmer_desc=programmer_desc5
    )
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}; output - {results5['exec_result']}")
    agents.append(f"Programmer agent {results5['programmer_agent'].id}, validating candidates, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}; exec_result: {results5['exec_result']}")
    logs.append(results5['subtask_desc'])

    valid_lists = results5['exec_result']

    if len(valid_lists) == 1:
        # Start true branch
        # End true branch (no additional processing needed)
        pass
    else:
        # Start false branch
        specific_format_instruction6 = "Subtask 6: Identify, clarify, and validate units: generate an appropriate message or corrective action for non-unique solution cases"
        specific_format_desc6 = {
            'instruction': specific_format_instruction6,
            'input': [taskInfo, valid_lists],
            'temperature': 0.0,
            'context': ["user query", "validation results"],
            'format': 'short and concise, without explanation'
        }
        results6 = await self.specific_format(
            subtask_id="subtask_6",
            formatter_desc=specific_format_desc6
        )
        answer_generate_instruction6 = "Subtask 6: Generate final message for non-unique or zero valid solution cases"
        answer_generate_desc6 = {
            'instruction': answer_generate_instruction6,
            'input': [taskInfo, results6['thinking'], results6['answer']],
            'temperature': 0.0,
            'context': ["user query", "clarification message"]
        }
        results6_final = await self.answer_generate(
            subtask_id="subtask_6_final",
            cot_agent_desc=answer_generate_desc6
        )
        sub_tasks.append(f"Subtask 6 output: thinking - {results6_final['thinking'].content}; answer - {results6_final['answer'].content}")
        agents.append(f"SpecificFormat agent {results6['formatter_agent'].id}, message generation, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
        logs.append(results6['subtask_desc'])
        # End false branch

    review_instruction7 = "Subtask 7: Validate and assess the final result presentation for correctness and completeness"
    review_desc7 = {
        'instruction': review_instruction7,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "validation results"]
    }
    results7 = await self.review(
        subtask_id="subtask_7",
        review_desc=review_desc7
    )
    sub_tasks.append(f"Subtask 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
    agents.append(f"Review agent {results7['review_agent'].id}, validating final result, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    specific_format_instruction8 = "Subtask 8: Enhance clarity and coherence of the decomposition and final output"
    specific_format_desc8 = {
        'instruction': specific_format_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "review feedback"],
        'format': 'clear and coherent'
    }
    results8 = await self.specific_format(
        subtask_id="subtask_8",
        formatter_desc=specific_format_desc8
    )
    sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    agents.append(f"SpecificFormat agent {results8['formatter_agent'].id}, enhancing clarity, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    specific_format_instruction9 = "Subtask 9: Format the verified artifact into the specified JSON structure"
    specific_format_desc9 = {
        'instruction': specific_format_instruction9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "clarified output"],
        'format': 'JSON structure as required'
    }
    results9 = await self.specific_format(
        subtask_id="subtask_9",
        formatter_desc=specific_format_desc9
    )
    answer_generate_instruction9 = "Subtask 9: Generate the final answer in the required format"
    answer_generate_desc9 = {
        'instruction': answer_generate_instruction9,
        'input': [taskInfo, results9['thinking'], results9['answer']],
        'temperature': 0.0,
        'context': ["user query", "formatted output"]
    }
    results_final = await self.answer_generate(
        subtask_id="subtask_10",
        cot_agent_desc=answer_generate_desc9
    )
    sub_tasks.append(f"Subtask 9 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    agents.append(f"AnswerGenerate agent {results_final['cot_agent'].id}, generating final answer, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs
