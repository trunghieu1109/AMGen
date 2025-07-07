async def forward_15(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction2 = "Subtask 2: Derive the inclusion-exclusion formula for four sets and introduce unknown x representing residents owning all four items, with context from taskInfo"
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_agent_desc2,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, deriving inclusion-exclusion formula, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction1 = "Subtask 1: Extract and define essential variables (counts of residents owning each item and combinations) from the problem statement, with context from taskInfo and output of subtask_2"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting variables, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction3 = "Subtask 3: Evaluate and characterize the given counts of single, double, and triple ownerships against the total population, using outputs from subtasks 1 and 2"
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, evaluating counts, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    x = None
    for i in range(1,4):
        if i == 1:
            programmer_instruction5 = "Subtask 5: Validate consistency of individual, pairwise, and triple counts against total residents using code"
            programmer_desc5 = {
                'instruction': programmer_instruction5,
                'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
                'temperature': 0.0,
                'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"],
                'entry_point': 'validate_counts'
            }
            results5 = await self.programmer(
                subtask_id="subtask_5",
                programmer_desc=programmer_desc5
            )
            agents.append(f"Programmer Agent {results5['programmer_agent'].id}, validating counts, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}, executing results: {results5['exec_result']}")
            sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}; output - {results5['exec_result']}")
            logs.append(results5['subtask_desc'])
        elif i == 2:
            aggregate_instruction4 = "Subtask 4: Consolidate validated counts into inclusion-exclusion equation structure"
            aggregate_desc4 = {
                'instruction': aggregate_instruction4,
                'input': [taskInfo, results5['thinking'], results5['answer']],
                'temperature': 0.0,
                'context': ["user query", "validated counts from subtask 5"]
            }
            results4 = await self.aggregate(
                subtask_id="subtask_4",
                aggregate_desc=aggregate_desc4
            )
            agents.append(f"CoT agent {results4['aggregate_agent'].id}, consolidating counts, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
            sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
            logs.append(results4['subtask_desc'])
        else:
            cot_sc_instruction3_3 = "Subtask 3: Generate candidate solution for x (residents owning all four) by solving the inclusion-exclusion equation"
            cot_sc_desc3_3 = {
                'instruction': cot_sc_instruction3_3,
                'input': [taskInfo, results4['thinking'], results4['answer']],
                'temperature': 0.5,
                'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
            }
            results6 = await self.sc_cot(
                subtask_id="subtask_3",
                cot_sc_desc=cot_sc_desc3_3,
                n_repeat=self.max_sc
            )
            for idx, key in enumerate(results6['list_thinking']):
                agents.append(f"CoT-SC agent {results6['cot_agent'][idx].id}, generating candidate x, thinking: {results6['list_thinking'][idx]}; answer: {results6['list_answer'][idx]}")
            sub_tasks.append(f"Subtask 3 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
            logs.append(results6['subtask_desc'])
            try:
                x = int(results6['answer'].content.strip())
            except:
                x = None

    if x is not None and isinstance(x, int) and x >= 0:
        formatter_instruction9 = "Subtask 9: Format the computed integer x into the required output format (integer only)"
        formatter_desc9 = {
            'instruction': formatter_instruction9,
            'input': [x],
            'temperature': 0.0,
            'context': ["user query"],
            'format': 'short and concise, without explanation'
        }
        results9 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=formatter_desc9
        )
        agents.append(f"CoT agent {results9['formatter_agent'].id}, formatting output, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        formatter_instruction8 = "Subtask 8: Enhance clarity and coherence of the final formatted output"
        formatter_desc8 = {
            'instruction': formatter_instruction8,
            'input': [results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=formatter_desc8
        )
        agents.append(f"CoT agent {results8['formatter_agent'].id}, enhancing clarity, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
        return final_answer, logs
    else:
        formatter_instruction9 = "Subtask 9: Format the invalid computed x result for error handling"
        formatter_desc9 = {
            'instruction': formatter_instruction9,
            'input': [x],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results9 = await self.specific_format(
            subtask_id="subtask_9",
            formatter_desc=formatter_desc9
        )
        agents.append(f"CoT agent {results9['formatter_agent'].id}, formatting invalid output, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
        sub_tasks.append(f"Subtask 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
        logs.append(results9['subtask_desc'])

        review_instruction7 = "Subtask 7: Validate that the computed x violates constraints and document the issue"
        review_desc7 = {
            'instruction': review_instruction7,
            'input': [taskInfo, results9['thinking'], results9['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
        }
        results7 = await self.review(
            subtask_id="subtask_7",
            review_desc=review_desc7
        )
        agents.append(f"Review agent {results7['review_agent'].id}, validating invalid x, feedback: {results7['thinking'].content}; correct: {results7['answer'].content}")
        sub_tasks.append(f"Subtask 7 output: feedback - {results7['thinking'].content}; correct - {results7['answer'].content}")
        logs.append(results7['subtask_desc'])

        formatter_instruction8 = "Subtask 8: Refine and clarify an error message or explanation for an invalid result"
        formatter_desc8 = {
            'instruction': formatter_instruction8,
            'input': [results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "feedback of subtask 7", "correct of subtask 7"]
        }
        results8 = await self.specific_format(
            subtask_id="subtask_8",
            formatter_desc=formatter_desc8
        )
        agents.append(f"CoT agent {results8['formatter_agent'].id}, refining error message, thinking: {results8['thinking'].content}; answer: {results8['answer'].content}")
        sub_tasks.append(f"Subtask 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
        logs.append(results8['subtask_desc'])

        cot_agent_instruction6 = "Subtask 6: Identify and clarify any ambiguities or errors if x is invalid"
        cot_agent_desc6 = {
            'instruction': cot_agent_instruction6,
            'input': [taskInfo, results8['thinking'], results8['answer'], results7['thinking'], results7['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8", "feedback of subtask 7", "correct of subtask 7"]
        }
        results6 = await self.answer_generate(
            subtask_id="subtask_6",
            cot_agent_desc=cot_agent_desc6
        )
        agents.append(f"CoT agent {results6['cot_agent'].id}, clarifying invalid x, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])

        final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
        return final_answer, logs
