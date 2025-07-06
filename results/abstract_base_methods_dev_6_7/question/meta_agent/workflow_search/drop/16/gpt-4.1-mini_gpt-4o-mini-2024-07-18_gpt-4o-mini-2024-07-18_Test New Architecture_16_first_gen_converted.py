async def forward_16(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction1 = "Subtask 1: Decompose the input passage to logically extract fertility rates for 1999 and the current period, then calculate the difference in fertility rate points."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, decomposing passage and calculating fertility rate difference, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    subtask1_thinking = results1['thinking']
    subtask1_answer = results1['answer']

    subtask_id = 2
    refined_thinking = subtask1_thinking
    refined_answer = subtask1_answer

    for i in range(self.max_round):
        cot_reflect_instruction = f"Subtask {subtask_id}: Iteratively evaluate and refine the calculation of the fertility rate difference to improve clarity, accuracy, and completeness based on previous output."
        critic_instruction = f"Subtask {subtask_id}: Review the refined fertility rate difference calculation and provide feedback on limitations or errors."
        cot_reflect_desc = {
            'instruction': cot_reflect_instruction,
            'input': [taskInfo, refined_thinking, refined_answer],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "previous thinking", "previous answer"]
        }
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_refine = await self.reflexion(
            subtask_id=f"subtask_{subtask_id}",
            cot_reflect_desc=cot_reflect_desc,
            critic_desc=critic_desc,
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results_refine['cot_agent'].id}, refining fertility rate difference, thinking: {results_refine['list_thinking'][0].content}; answer: {results_refine['list_answer'][0].content}")
        for k in range(min(1, len(results_refine['list_feedback']))):
            agents.append(f"Critic agent {results_refine['critic_agent'].id}, providing feedback, thinking: {results_refine['list_feedback'][k].content}; answer: {results_refine['list_correct'][k].content}")
        sub_tasks.append(f"Subtask {subtask_id} output: thinking - {results_refine['list_thinking'][0].content}; answer - {results_refine['list_answer'][0].content}")
        logs.append(results_refine['subtask_desc'])
        refined_thinking = results_refine['list_thinking'][0]
        refined_answer = results_refine['list_answer'][0]
        subtask_id += 1

    review_instruction = f"Subtask {subtask_id}: Optionally evaluate the calculated fertility rate difference for correctness and reliability against the passage data."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, refined_thinking, refined_answer],
        'temperature': 0.0,
        'context': ["user query", "refined thinking", "refined answer"]
    }
    results_review = await self.review(
        subtask_id=f"subtask_{subtask_id}",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing refined fertility rate difference, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Subtask {subtask_id} output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])

    subtask_id += 1

    aggregate_instruction = f"Subtask {subtask_id}: Aggregate the outputs from validation and reasoning steps and select the most coherent and consistent final answer to the fertility rate difference question."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo, refined_thinking, refined_answer, results_review['feedback'], results_review['correct']],
        'temperature': 0.0,
        'context': ["user query", "refined answer", "review feedback"]
    }
    results_aggregate = await self.aggregate(
        subtask_id=f"subtask_{subtask_id}",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, aggregating and selecting best final answer, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Subtask {subtask_id} output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])

    final_answer = await self.make_final_answer(results_aggregate['thinking'], results_aggregate['answer'], sub_tasks, agents)
    return final_answer, logs
