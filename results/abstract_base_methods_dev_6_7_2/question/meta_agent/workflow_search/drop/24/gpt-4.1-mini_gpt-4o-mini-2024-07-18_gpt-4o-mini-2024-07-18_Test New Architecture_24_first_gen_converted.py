async def forward_24(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    
    # Stage 0: Construct Logical Reasoning Sequence
    cot_instruction1 = "Subtask 1: Decompose the passage information into an ordered sequence of logical steps to identify the largest group among Ethiopians, Cuban soldiers, and Soviet advisors."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing passage, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    # Control Flow 1: start_loop
    loop_iterations = 3
    refined_thinkings = []
    refined_answers = []
    for i in range(loop_iterations):
        reflexion_instruction = f"Subtask 2: Iteratively evaluate and revise the initial reasoning and answer from iteration {i+1} to enhance clarity, consistency, and completeness regarding the largest group identification."
        revise_desc = {
            'instruction': reflexion_instruction,
            'input': [taskInfo, results1['thinking'], results1['answer']] + refined_thinkings + refined_answers,
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"] + [f"thinking of iteration {j+1}" for j in range(i)] + [f"answer of iteration {j+1}" for j in range(i)]
        }
        results2 = await self.reflexion(
            subtask_id=f"subtask_2_iter_{i+1}",
            cot_reflect_desc=revise_desc,
            critic_desc={
                'instruction': f"Please review the revised reasoning and answer at iteration {i+1} and provide feedback.",
                'output': ["feedback", "correct"],
                'temperature': 0.0
            },
            n_repeat=self.max_round
        )
        agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, iteration {i+1}, thinking: {results2['list_thinking'][0].content}; answer: {results2['list_answer'][0].content}")
        for k in range(min(self.max_round, len(results2['list_feedback']))):
            agents.append(f"Critic agent {results2['critic_agent'].id}, iteration {i+1}, feedback: {results2['list_feedback'][k].content}; correct: {results2['list_correct'][k].content}")
            if k + 1 < len(results2['list_thinking']) and k + 1 < len(results2['list_answer']):
                agents.append(f"Reflexion CoT agent {results2['cot_agent'].id}, iteration {i+1}, refining final answer, thinking: {results2['list_thinking'][k + 1].content}; answer: {results2['list_answer'][k + 1].content}")
        refined_thinkings.append(results2['thinking'])
        refined_answers.append(results2['answer'])
        sub_tasks.append(f"Subtask 2 iteration {i+1} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
    
    # Control Flow 1_end
    
    # Stage 2: Consolidate and select optimal output
    aggregate_instruction = "Subtask 3: Aggregate multiple variant outputs from iterative improvements and select the most coherent and consistent final result for the largest group identification."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + refined_thinkings + refined_answers,
        'temperature': 0.0,
        'context': ["user query", "iterative refined answers"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    # Stage 3: Validate Output
    review_instruction = "Subtask 4: Evaluate the final aggregated output against correctness and reliability criteria to validate the answer identifying the largest group."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregated answer"]
    }
    results4 = await self.review(
        subtask_id="subtask_4",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results4['review_agent'].id}, feedback: {results4['feedback'].content}; correct: {results4['correct'].content}")
    sub_tasks.append(f"Subtask 4 output: feedback - {results4['feedback'].content}; correct - {results4['correct'].content}")
    logs.append(results4['subtask_desc'])
    
    reflexion_instruction4 = "Subtask 5: Based on the review feedback, refine and finalize the answer identifying the largest group."
    reflexion_desc4 = {
        'instruction': reflexion_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['feedback'], results4['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "aggregated answer", "review feedback"]
    }
    critic_desc4 = {
        'instruction': "Please review the final refined answer and provide any limitations.",
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=reflexion_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, final refinement thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, final feedback: {results5['list_feedback'][i].content}; correct: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    # Control Flow 0_end
    
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
