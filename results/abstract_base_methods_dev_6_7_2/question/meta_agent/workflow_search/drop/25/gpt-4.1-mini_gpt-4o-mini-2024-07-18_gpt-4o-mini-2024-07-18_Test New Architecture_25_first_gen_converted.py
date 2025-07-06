async def forward_25(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose the passage to identify all touchdown passes and their yardages, then determine the longest touchdown pass yardage as an initial answer."
    cot_agent_desc = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing passage for touchdown passes, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Control Flow 1: start_loop for iterative quality enhancement
    revised_thinking = results1['thinking']
    revised_answer = results1['answer']
    max_iterations = self.max_round if hasattr(self, 'max_round') else 3
    for i in range(max_iterations):
        cot_reflect_instruction2 = "Subtask 2: Evaluate the initial answer for clarity, consistency, and completeness; revise the reasoning and answer to improve quality."
        critic_instruction2 = "Please review the revised answer for any limitations or inconsistencies."
        cot_reflect_desc2 = {
            'instruction': cot_reflect_instruction2,
            'input': [taskInfo, revised_thinking, revised_answer],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "previous thinking", "previous answer"]
        }
        critic_desc2 = {
            'instruction': critic_instruction2,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results2 = await self.reflexion(
            subtask_id="subtask_2",
            cot_reflect_desc=cot_reflect_desc2,
            critic_desc=critic_desc2,
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, revising answer, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
        agents.append(f"Critic agent {results2['critic_agent'].id}, feedback: {results2['list_feedback'][0].content}; correction: {results2['list_correct'][0].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results2['list_thinking'][0].content}; revised_answer - {results2['list_answer'][0].content}")
        logs.append(results2['subtask_desc'])
        revised_thinking = results2['list_thinking'][0]
        revised_answer = results2['list_answer'][0]
    
    # Stage 2: Consolidate and select optimal output
    aggregate_instruction3 = "Subtask 3: Aggregate the revised answers and select the most coherent and consistent final longest touchdown pass yardage."
    aggregate_desc3 = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo, revised_thinking, revised_answer],
        'temperature': 0.0,
        'context': ["user query", "revised answers"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc3
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating revised answers, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 3: Validate Output (optional)
    review_instruction4 = "Subtask 4: Optionally validate the final answer against the passage to ensure correctness and reliability."
    review_desc4 = {
        'instruction': review_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "final answer"]
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc4
    )
    agents.append(f"Review agent {results4['review_agent'].id}, reviewing final answer, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
    sub_tasks.append(f"Subtask 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    
    final_answer = await self.make_final_answer(results3['thinking'], results3['answer'], sub_tasks, agents)
    return final_answer, logs