async def forward_42(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    for i in range(3):
        # Stage 1: Iterative Quality Enhancement
        # Subtask 1: Iteratively evaluate and refine the understanding of the census data and question to improve clarity, consistency, and completeness of the reasoning artifact.
        cot_reflect_instruction = f"Subtask {i+1}: Iteratively evaluate and refine the understanding of the census data and question to improve clarity, consistency, and completeness of the reasoning artifact. Iteration {i+1}."
        critic_instruction = f"Subtask {i+1}: Review the refinement and provide feedback for iteration {i+1}."
        cot_reflect_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo],
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
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, iteration {i+1}, feedback: {results_reflexion['list_feedback'][k].content}; correct: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteration {i+1}, refining answer, thinking: {results_reflexion['list_thinking'][k+1].content}; answer: {results_reflexion['list_answer'][k+1].content}")
        sub_tasks.append(f"Subtask {i+1} output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
    # Control Flow 2: end_loop
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction = "Subtask 4: Decompose the census data and question into an ordered logical reasoning sequence to determine which group is smaller: Macedonians or Albanians."
    cot_agent_desc = {
        'instruction': cot_instruction,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results_cot = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results_cot['cot_agent'].id}, analyzing census data, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
    logs.append(results_cot['subtask_desc'])
    
    # Stage 2: Consolidate and select optimal output (optional)
    aggregate_instruction = "Subtask 5: Aggregate multiple reasoning outputs and select the most coherent and consistent final result for the smaller group between Macedonians and Albanians."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer']] + [results_reflexion['answer'] for _ in range(3)],
        'temperature': 0.0,
        'context': ["user query", "solutions generated from previous subtasks"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    
    # Stage 3: Validate Output (optional)
    review_instruction = "Subtask 6: Evaluate the final answer against correctness and reliability criteria to ensure the output is accurate and complete."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context': ["user query", "final aggregated answer"]
    }
    results_review = await self.review(
        subtask_id="subtask_6",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Subtask 6 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_aggregate['thinking'], results_aggregate['answer'], sub_tasks, agents)
    return final_answer, logs