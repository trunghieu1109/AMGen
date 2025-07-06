async def forward_17(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction1 = "Subtask 1: Identify the number of fish Jen has from the task information."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying number of fish, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_instruction2 = "Subtask 2: Identify the daily food cost per fish from the task information."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, identifying daily food cost per fish, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    cot_sc_instruction3 = "Subtask 3: Generate candidate daily food cost outputs for one fish based on identified daily food cost per fish, considering possible interpretations or rounding."
    N = self.max_sc
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=N
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, generating candidate daily food cost, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    aggregate_instruction4 = "Subtask 4: Consolidate daily food cost outputs for all fish into total daily cost by multiplying number of fish and daily food cost per fish candidates."
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo, results1['answer'], results3['list_answer']],
        'temperature': 0.0,
        'context': ["user query", "candidate daily food costs", "number of fish"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating total daily food cost, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    review_instruction5 = "Subtask 5: Validate the consolidated total daily food cost for accuracy and completeness."
    review_desc5 = {
        'instruction': review_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc5
    )
    agents.append(f"Review agent {results5['review_agent'].id}, validating total daily food cost, feedback: {results5['thinking'].content}; correct: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: feedback - {results5['thinking'].content}; correct - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    cot_instruction6 = "Subtask 6: Determine the number of days in May."
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, determining number of days in May, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    debate_instruction7 = "Subtask 7: Calculate total monthly food cost by multiplying total daily cost by number of days in May."
    final_decision_instruction7 = "Subtask 7: Make final decision on total monthly food cost calculation."
    debate_desc7 = {
        'instruction': debate_instruction7,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5", "thinking of subtask 6", "answer of subtask 6"],
        'input': [taskInfo, results5['thinking'], results5['answer'], results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc7 = {
        'instruction': final_decision_instruction7,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results7 = await self.debate(
        subtask_id="subtask_7",
        debate_desc=debate_desc7,
        final_decision_desc=final_decision_desc7,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results7['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, calculating total monthly food cost, thinking: {results7['list_thinking'][round][idx].content}; answer: {results7['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating total monthly food cost, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    review_instruction8 = "Subtask 8: Validate the final monthly food cost calculation for correctness and completeness."
    review_desc8 = {
        'instruction': review_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    results8 = await self.review(
        subtask_id="subtask_8",
        review_desc=review_desc8
    )
    agents.append(f"Review agent {results8['review_agent'].id}, validating final monthly food cost, feedback: {results8['thinking'].content}; correct: {results8['answer'].content}")
    sub_tasks.append(f"Subtask 8 output: feedback - {results8['thinking'].content}; correct - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])
    
    final_answer = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer, logs
