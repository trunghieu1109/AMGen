async def forward_47(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Control Flow 1: start_loop
    for i in range(3):
        # Stage 1: Iterative Quality Enhancement
        # Subtask 1: Iteratively evaluate and refine the understanding of the passage and question to improve clarity, consistency, and completeness of extracted information.
        cot_reflect_instruction = "Subtask 1: Iteratively evaluate and refine the understanding of the passage and question to improve clarity, consistency, and completeness of extracted information."
        critic_instruction = "Please review the refined understanding and provide feedback on clarity, consistency, and completeness."
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
            subtask_id="subtask_1", 
            cot_reflect_desc=cot_reflect_desc, 
            critic_desc=critic_desc, 
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, iteratively refining understanding, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_reflexion['list_feedback']))):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, providing feedback, thinking: {results_reflexion['list_feedback'][k].content}; answer: {results_reflexion['list_correct'][k].content}")
            if k + 1 < len(results_reflexion['list_thinking']) and k + 1 < len(results_reflexion['list_answer']):
                agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, refining final answer, thinking: {results_reflexion['list_thinking'][k + 1].content}; answer: {results_reflexion['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task 1 output: thinking - {results_reflexion['thinking'].content}; answer - {results_reflexion['answer'].content}")
        logs.append(results_reflexion['subtask_desc'])
    # Control Flow 2: end_loop
    
    # Stage 0: Construct Logical Reasoning Sequence
    # Subtask 2: Decompose the passage and question into an ordered logical reasoning sequence to identify the second touchdown and determine its yardage.
    cot_instruction2 = "Subtask 2: Decompose the passage and question into an ordered logical reasoning sequence to identify the second touchdown and determine its yardage."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results_reflexion['thinking'], results_reflexion['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results_cot = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results_cot['cot_agent'].id}, decomposing passage and question, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
    logs.append(results_cot['subtask_desc'])
    
    # Stage 3: Validate Output (optional)
    # Subtask 3: Optionally evaluate the constructed answer for correctness and reliability against the passage and question criteria.
    review_instruction3 = "Subtask 3: Review the constructed answer for correctness and reliability against the passage and question criteria."
    review_desc3 = {
        'instruction': review_instruction3,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results_review = await self.review(
        subtask_id="subtask_3",
        review_desc=review_desc3
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing constructed answer, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Sub-task 3 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])
    
    cot_reflect_instruction3 = "Subtask 4: Based on the review feedback, filter and refine the answer to ensure correctness and reliability."
    critic_instruction3 = "Please review the filtered and refined answer and provide any limitations or further improvements."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer'], results_review['feedback'], results_review['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "feedback of subtask 3", "correct of subtask 3"]
    }
    critic_desc3 = {
        'instruction': critic_instruction3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results_refine = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_refine['cot_agent'].id}, refining answer based on review, thinking: {results_refine['list_thinking'][0].content}; answer: {results_refine['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results_refine['list_feedback']))):
        agents.append(f"Critic agent {results_refine['critic_agent'].id}, providing feedback, thinking: {results_refine['list_feedback'][k].content}; answer: {results_refine['list_correct'][k].content}")
        if k + 1 < len(results_refine['list_thinking']) and k + 1 < len(results_refine['list_answer']):
            agents.append(f"Reflexion CoT agent {results_refine['cot_agent'].id}, refining final answer, thinking: {results_refine['list_thinking'][k + 1].content}; answer: {results_refine['list_answer'][k + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_refine['thinking'].content}; answer - {results_refine['answer'].content}")
    logs.append(results_refine['subtask_desc'])
    
    # Stage 2: Consolidate and select optimal output
    # Subtask 5: Aggregate multiple variant outputs and select the most coherent and consistent final result.
    aggregate_instruction5 = "Subtask 5: Aggregate the refined answers and select the most coherent and consistent final result regarding the yardage of the second touchdown."
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer'], results_refine['thinking'], results_refine['answer']],
        'temperature': 0.0,
        'context': ["user query", "solutions generated from subtask 2 and 4"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc5
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, aggregating refined answers, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_aggregate['thinking'], results_aggregate['answer'], sub_tasks, agents)
    return final_answer, logs
