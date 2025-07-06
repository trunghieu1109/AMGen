async def forward_23(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    refined_outputs = []
    loop_iterations = 3
    for i in range(loop_iterations):
        reflexion_instruction = f"Subtask {i+1}: Iteratively evaluate and refine the extraction and interpretation of racial group data from the passage to improve clarity, consistency, and completeness. Iteration {i+1}."
        critic_instruction = f"Please review the refined racial group data from iteration {i+1} and provide limitations."
        reflexion_desc = {
            'instruction': reflexion_instruction,
            'input': [taskInfo] + refined_outputs,
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query"]
        }
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion = await self.reflexion(
            subtask_id=f"subtask_{i+1}",
            cot_reflect_desc=reflexion_desc,
            critic_desc=critic_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][k].content}; correct: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining, thinking: {results_reflexion['list_thinking'][k + 1].content}; answer: {results_reflexion['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
        revise_instruction = f"Subtask {i+1} revise: Revise previous refined racial group data to improve clarity and consistency."
        revise_desc = {
            'instruction': revise_instruction,
            'input': [taskInfo, results_reflexion['thinking'], results_reflexion['answer']],
            'temperature': 0.0,
            'context': ['user query', 'thinking of reflexion', 'answer of reflexion']
        }
        results_revise = await self.revise(
            subtask_id=f"subtask_{i+1}_revise",
            revise_desc=revise_desc
        )
        agents.append(f"Revise agent {results_revise['revise_agent'].id}, iteration {i+1}, feedback: [feedback], thinking: {results_revise['thinking'].content}; revised_solution: {results_revise['revised_solution'].content}")
        sub_tasks.append(f"Sub-task {i+1} revise output: thinking - {results_revise['thinking'].content}; revised_solution - {results_revise['revised_solution'].content}")
        logs.append(results_revise['subtask_desc'])
        refined_outputs = [results_revise['revised_solution']]
    cot_instruction = "Sub-task 4: Decompose the refined racial group data into an ordered logical sequence to identify which racial group is the second smallest."
    cot_agent_desc = {
        'instruction': cot_instruction,
        'input': [taskInfo, refined_outputs[0]],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_cot = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results_cot['cot_agent'].id}, decomposing refined data, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
    logs.append(results_cot['subtask_desc'])
    review_instruction = "Sub-task 5: Evaluate the identified second smallest racial group against correctness and reliability criteria to validate the output."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer']],
        'temperature': 0.0,
        'context': ['user query', 'thinking of subtask 4', 'answer of subtask 4']
    }
    results_review = await self.review(
        subtask_id="subtask_5",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing subtask 4 output, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Sub-task 5 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])
    aggregate_instruction = "Sub-task 6: Aggregate multiple validation outputs and select the most coherent and consistent final result."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer'], results_review['feedback'], results_review['correct']],
        'temperature': 0.0,
        'context': ['user query', 'solutions generated from subtask 4 and 5']
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_6",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, aggregating validation outputs, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    final_answer = await self.make_final_answer(results_aggregate['thinking'], results_aggregate['answer'], sub_tasks, agents)
    return final_answer, logs