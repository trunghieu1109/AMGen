async def forward_26(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_reflect_instruction0 = "Sub-task 1: Reflect on the formula that the number of sets B equals sum of 2^{a-1} for a in A and validate its correctness against the problem statement."
    critic_instruction0 = "Please review the reasoning about the formula and identify any incorrect assumptions or errors."
    cot_reflect_desc0 = {
        'instruction': cot_reflect_instruction0,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc0 = {
        'instruction': critic_instruction0,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results0 = await self.reflexion(
        subtask_id="subtask_1_reflect_formula_correctness",
        cot_reflect_desc=cot_reflect_desc0,
        critic_desc=critic_desc0,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results0['cot_agent'].id}, reflecting on formula correctness, thinking: {results0['list_thinking'][0].content}; answer: {results0['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results0['list_feedback']))):
        agents.append(f"Critic agent {results0['critic_agent'].id}, feedback: {results0['list_feedback'][i].content}; correct: {results0['list_correct'][i].content}")
        if i + 1 < len(results0['list_thinking']) and i + 1 < len(results0['list_answer']):
            agents.append(f"Reflexion CoT agent {results0['cot_agent'].id}, refining, thinking: {results0['list_thinking'][i+1].content}; answer: {results0['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])
    
    cot_instruction1 = "Sub-task 2: Convert 2024 to binary and identify all positive integers a such that 2^{a-1} corresponds to the '1' bits in the binary representation, thus candidate elements of set A."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.0,
        'context': ["user query", "reflection on formula correctness"]
    }
    results1 = await self.cot(
        subtask_id="subtask_2_identify_candidate_elements",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, identifying candidate elements from binary decomposition, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_reflect_instruction2 = "Sub-task 3: Verify that the sum of 2^{a-1} for the candidate set A equals exactly 2024, and reflect on the correctness of the candidate set."
    critic_instruction2 = "Please review the candidate set A and confirm if the sum of 2^{a-1} equals 2024, providing feedback on any discrepancies."
    cot_reflect_desc2 = {
        'instruction': cot_reflect_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "candidate elements identification"]
    }
    critic_desc2 = {
        'instruction': critic_instruction2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results2 = await self.reflexion(
        subtask_id="subtask_3_verify_candidate_set",
        cot_reflect_desc=cot_reflect_desc2,
        critic_desc=critic_desc2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, verifying candidate set sum, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results2['list_feedback']))):
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][i].content}; correct: {results2['list_correct'][i].content}")
        if i + 1 < len(results2['list_thinking']) and i + 1 < len(results2['list_answer']):
            agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, refining, thinking: {results2['list_thinking'][i+1].content}; answer: {results2['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    cot_sc_instruction3 = "Sub-task 4: Generate multiple candidate sets A that satisfy sum of 2^{a-1} = 2024 by exploring different decompositions, and cross-validate these candidates for correctness."
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "verification of candidate set"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_4_generate_and_crossvalidate_candidates",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, candidate set generation and validation, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
        sub_tasks.append(f"Sub-task 4 output iteration {idx+1}: thinking - {results3['list_thinking'][idx]}; answer - {results3['list_answer'][idx]}")
    logs.append(results3['subtask_desc'])
    
    aggregate_instruction4 = "Sub-task 5: Aggregate all candidate sets generated to identify the most consistent and frequent solution for set A."
    aggregate_desc4 = {
        'instruction': aggregate_instruction4,
        'input': [taskInfo] + results3['list_answer'],
        'temperature': 0.0,
        'context': ["user query", "candidate sets from subtask 4"]
    }
    results4 = await self.aggregate(
        subtask_id="subtask_5_aggregate_candidates",
        aggregate_desc=aggregate_desc4
    )
    agents.append(f"Aggregate agent {results4['aggregate_agent'].id}, consolidating candidate sets, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    cot_instruction5 = "Sub-task 6: Check for any missing or overlapping cases among the aggregated candidate sets and fill gaps to ensure the solution is complete and unique."
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregation of candidate sets"]
    }
    results5 = await self.cot(
        subtask_id="subtask_6_check_completeness",
        cot_agent_desc=cot_agent_desc5
    )
    agents.append(f"CoT agent {results5['cot_agent'].id}, checking completeness and uniqueness, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    unique_solution = False
    if "unique" in results5['answer'].content.lower() or "one" in results5['answer'].content.lower():
        unique_solution = True
    
    if not unique_solution:
        cot_reflect_instruction6 = "Sub-task 7: Refine ambiguous candidate sets and validate that the sum of 2^{a-1} equals 2024 exactly, confirming the unique solution set A."
        critic_instruction6 = "Review the refined candidate set and confirm correctness and uniqueness."
        cot_reflect_desc6 = {
            'instruction': cot_reflect_instruction6,
            'input': [taskInfo, results5['thinking'], results5['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "checking completeness"]
        }
        critic_desc6 = {
            'instruction': critic_instruction6,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results6 = await self.reflexion(
            subtask_id="subtask_7_refine_and_validate_final_candidate",
            cot_reflect_desc=cot_reflect_desc6,
            critic_desc=critic_desc6,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining and validating final candidate, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
        for i in range(min(self.max_round, len(results6['list_feedback']))):
            agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][i].content}; correct: {results6['list_correct'][i].content}")
            if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
                agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining, thinking: {results6['list_thinking'][i+1].content}; answer: {results6['list_answer'][i+1].content}")
        sub_tasks.append(f"Sub-task 7 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])
        final_candidate_thinking = results6['thinking']
        final_candidate_answer = results6['answer']
    else:
        final_candidate_thinking = results5['thinking']
        final_candidate_answer = results5['answer']
    
    cot_instruction7 = "Sub-task 8: Format the identified unique solution set A clearly and concisely for presentation."
    cot_agent_desc7 = {
        'instruction': cot_instruction7,
        'input': [taskInfo, final_candidate_thinking, final_candidate_answer],
        'temperature': 0.0,
        'context': ["user query", "final candidate set"]
    }
    results7 = await self.cot(
        subtask_id="subtask_8_format_solution",
        cot_agent_desc=cot_agent_desc7
    )
    agents.append(f"CoT agent {results7['cot_agent'].id}, formatting solution, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    review_instruction8 = "Sub-task 9: Verify the final solution's correctness and completeness, ensuring sum of 2^{a-1} equals 2024 and sum of elements of A is correct."
    review_desc8 = {
        'instruction': review_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "formatted solution"]
    }
    results8 = await self.review(
        subtask_id="subtask_9_verify_final_solution",
        review_desc=review_desc8
    )
    agents.append(f"Review agent {results8['review_agent'].id}, verifying final solution, feedback: {results8['thinking'].content}; correct: {results8['answer'].content}")
    sub_tasks.append(f"Sub-task 9 output: feedback - {results8['thinking'].content}; correct - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])
    
    cot_instruction9 = "Sub-task 10: Produce the final numeric answer, the sum of the elements of set A, in a short and concise numeric format."
    cot_agent_desc9 = {
        'instruction': cot_instruction9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "verification feedback"],
        'format': 'short and concise, numeric only'
    }
    results9 = await self.cot(
        subtask_id="subtask_10_produce_final_numeric_answer",
        cot_agent_desc=cot_agent_desc9
    )
    agents.append(f"CoT agent {results9['cot_agent'].id}, producing final numeric answer, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])
    
    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
