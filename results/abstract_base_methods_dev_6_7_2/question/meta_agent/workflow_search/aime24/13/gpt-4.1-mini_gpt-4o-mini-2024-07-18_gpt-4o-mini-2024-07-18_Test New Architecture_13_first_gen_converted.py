async def forward_13(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_sc_instruction1 = "Subtask 1: Identify all elements and configurations of tangent circles and triangle ABC that satisfy the given constraints, including the radius relations and tangency conditions."
    cot_sc_desc1 = {
        'instruction': cot_sc_instruction1,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results1['list_thinking']):
        agents.append(f"SC-CoT agent {results1['cot_agent'][idx].id}, identifying constrained elements, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    candidate_thinkings = []
    candidate_answers = []
    for i in range(self.max_sc):
        cot_instruction2 = f"Subtask 2.{i+1}: Generate candidate outputs for the inradius of triangle ABC based on the identified constraints and circle arrangements from Subtask 1, applying logical reasoning steps."
        cot_desc2 = {
            'instruction': cot_instruction2,
            'input': [taskInfo, results1['list_thinking'][i], results1['list_answer'][i]],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2 = await self.cot(
            subtask_id=f"subtask_2_{i+1}",
            cot_agent_desc=cot_desc2
        )
        agents.append(f"CoT agent {results2['cot_agent'].id}, generating candidate output {i+1}, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Subtask 2.{i+1} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        candidate_thinkings.append(results2['thinking'].content)
        candidate_answers.append(results2['answer'].content)
    
    aggregate_instruction3 = "Subtask 3: Aggregate the candidate inradius values generated in Subtask 2, evaluate their consistency and coherence, and select the most plausible and coherent inradius value."
    aggregate_desc3 = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo] + candidate_answers,
        'temperature': 0.0,
        'context': ["user query", "candidate solutions from subtask 2"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc3
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating candidate inradius values, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    review_instruction4 = "Subtask 4: Independently review and validate the aggregated inradius value for correctness, completeness, and consistency with the problem constraints."
    review_desc4 = {
        'instruction': review_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregated solution from subtask 3"]
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc4
    )
    agents.append(f"Review agent {results4['review_agent'].id}, reviewing aggregated inradius, feedback: {results4['thinking'].content}; correct: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: feedback - {results4['thinking'].content}; correct - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    condition = results4['answer'].content.strip().lower() == 'correct'
    if condition:
        cot_reflect_instruction5 = "Subtask 5: Identify and fill any gaps or missing elements in the solution artifact to ensure completeness and clarity."
        cot_reflect_desc5 = {
            'instruction': cot_reflect_instruction5,
            'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "aggregated solution", "review feedback"]
        }
        results5 = await self.answer_generate(
            subtask_id="subtask_5",
            cot_agent_desc=cot_reflect_desc5
        )
        agents.append(f"AnswerGenerate agent {results5['cot_agent'].id}, filling gaps in solution, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
        sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
        logs.append(results5['subtask_desc'])
        final_thinking = results5['thinking']
        final_answer = results5['answer']
    else:
        cot_reflect_instruction6 = "Subtask 6: Refine and clarify the solution based on review feedback to improve accuracy and completeness."
        cot_reflect_desc6 = {
            'instruction': cot_reflect_instruction6,
            'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "aggregated solution", "review feedback"]
        }
        results6 = await self.answer_generate(
            subtask_id="subtask_6",
            cot_agent_desc=cot_reflect_desc6
        )
        agents.append(f"AnswerGenerate agent {results6['cot_agent'].id}, refining solution after review, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
        sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
        logs.append(results6['subtask_desc'])
        final_thinking = results6['thinking']
        final_answer = results6['answer']
    
    formatter_instruction7 = "Subtask 7: Format the final inradius answer as a reduced fraction m/n and compute m+n as the final numeric answer."
    formatter_desc7 = {
        'instruction': formatter_instruction7,
        'input': [taskInfo, final_thinking.content, final_answer.content],
        'temperature': 0.0,
        'context': ["user query", "final thinking", "final answer"],
        'format': 'short and concise, return only integer m+n'
    }
    results7 = await self.specific_format(
        subtask_id="subtask_7",
        formatter_desc=formatter_desc7
    )
    agents.append(f"SpecificFormat agent {results7['formatter_agent'].id}, formatting final numeric answer m+n, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Subtask 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])
    
    final_answer_processed = await self.make_final_answer(results7['thinking'], results7['answer'], sub_tasks, agents)
    return final_answer_processed, logs