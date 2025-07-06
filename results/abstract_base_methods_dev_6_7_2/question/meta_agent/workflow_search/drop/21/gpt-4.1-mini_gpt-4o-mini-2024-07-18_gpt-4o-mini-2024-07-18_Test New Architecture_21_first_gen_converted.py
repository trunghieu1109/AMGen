async def forward_21(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start sequential
    
    # Control Flow 1: start loop
    loop_iterations = 3
    refined_outputs = []
    for i in range(loop_iterations):
        
        # Stage 1: Iterative Quality Enhancement
        # Subtask 1: Iteratively evaluate and modify the extraction of Peyton Manning's passing statistics from the passage to enhance clarity, consistency, and completeness.
        cot_reflect_instruction = f"Sub-task 1: Iteratively evaluate and revise the extraction of Peyton Manning's passing statistics from the passage to improve clarity, consistency, and completeness. Iteration {i+1}."
        revise_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo],
            'temperature': 0.0,
            'context': ["user query"]
        }
        results_refine = await self.reflexion(
            subtask_id=f"subtask_{2*i+1}",
            cot_reflect_desc=revise_desc,
            critic_desc={
                'instruction': "Please review the revised extraction and provide feedback and correctness.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results_refine['cot_agent'].id}, iteration {i+1}, thinking: {results_refine['list_thinking'][0].content}; answer: {results_refine['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results_refine['list_feedback']))):
            agents.append(f"Critic agent {results_refine['critic_agent'].id}, iteration {i+1}, feedback: {results_refine['list_feedback'][k].content}; correct: {results_refine['list_correct'][k].content}")
            if k + 1 < len(results_refine['list_thinking']) and k + 1 < len(results_refine['list_answer']):
                agents.append(f"Reflexion CoT agent {results_refine['cot_agent'].id}, iteration {i+1}, refining, thinking: {results_refine['list_thinking'][k + 1].content}; answer: {results_refine['list_answer'][k + 1].content}")
        sub_tasks.append(f"Sub-task {2*i+1} output: thinking - {results_refine['thinking'].content}; answer - {results_refine['answer'].content}")
        logs.append(results_refine['subtask_desc'])
        refined_outputs.append(results_refine['answer'].content)
        
        # Stage 0: Construct Logical Reasoning Sequence
        # Subtask 2: Decompose the passage information into an ordered logical sequence to determine the number of passes Peyton Manning threw.
        cot_instruction = f"Sub-task 2: Decompose the passage information into an ordered logical sequence to determine the number of passes Peyton Manning threw, iteration {i+1}."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo, results_refine['thinking'], results_refine['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results_cot = await self.cot(
            subtask_id=f"subtask_{2*i+2}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results_cot['cot_agent'].id}, iteration {i+1}, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
        sub_tasks.append(f"Sub-task {2*i+2} output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
        logs.append(results_cot['subtask_desc'])
        refined_outputs.append(results_cot['answer'].content)
    
    # Control Flow 2: end loop
    
    # Stage 3: Validate Output (optional)
    # Subtask 3: Evaluate the derived answer for correctness and reliability by reviewing the reasoning and data extraction.
    review_instruction = "Sub-task 3: Evaluate the derived answer for correctness and reliability by reviewing the reasoning and data extraction."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo] + refined_outputs,
        'temperature': 0.0,
        'context': ["user query", "refined outputs"]
    }
    results_review = await self.review(
        subtask_id="subtask_3",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Sub-task 3 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])
    
    # Stage 2: Consolidate and select optimal output
    # Subtask 4: Aggregate the reviewed outputs and select the most coherent and consistent final answer for the number of passes Peyton Manning threw.
    aggregate_instruction = "Sub-task 4: Aggregate multiple variant outputs and select the most coherent and consistent final result for the number of passes Peyton Manning threw."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + refined_outputs + [results_review['feedback'], results_review['correct']],
        'temperature': 0.0,
        'context': ["user query", "refined outputs", "review feedback"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_4",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_aggregate['thinking'], results_aggregate['answer'], sub_tasks, agents)
    return final_answer, logs