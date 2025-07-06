async def forward_37(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start_sequential
    # Stage 1: Iterative Quality Enhancement (loop)
    improved_thinking_list = []
    improved_answer_list = []
    
    # Control Flow 1: start_loop
    for i in range(self.max_round):
        if i == 0:
            # Stage 0: Construct Logical Reasoning Sequence - subtask_1 (CoT | AnswerGenerate)
            cot_instruction = "Subtask 1: Decompose the passage and question into a logical reasoning sequence to derive an initial answer identifying who did not join the alliance at Polin's request."
            cot_agent_desc = {
                'instruction': cot_instruction,
                'input': [taskInfo],
                'temperature': 0.0,
                'context': ["user query"]
            }
            results_cot = await self.cot(
                subtask_id="subtask_1",
                cot_agent_desc=cot_agent_desc
            )
            agents.append(f"CoT agent {results_cot['cot_agent'].id}, analyzing passage and question, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
            sub_tasks.append(f"Subtask 1 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
            logs.append(results_cot['subtask_desc'])
            thinking_to_improve = results_cot['thinking']
            answer_to_improve = results_cot['answer']
        else:
            thinking_to_improve = improved_thinking_list[-1]
            answer_to_improve = improved_answer_list[-1]
        # Stage 1: Iterative Quality Enhancement - subtask_2 (Reflexion | Revise)
        reflexion_instruction = "Subtask 2: Iteratively evaluate and improve the initial reasoning and answer for clarity, consistency, and completeness regarding who did not join the alliance at Polin's request."
        critic_instruction = "Please review the reasoning and answer for limitations and suggest improvements."
        reflexion_desc = {
            'instruction': reflexion_instruction,
            'input': [taskInfo, thinking_to_improve, answer_to_improve],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "previous thinking", "previous answer"]
        }
        critic_desc = {
            'instruction': critic_instruction,
            'output': ["feedback", "correct"],
            'temperature': 0.0
        }
        results_reflexion = await self.reflexion(
            subtask_id="subtask_2",
            cot_reflect_desc=reflexion_desc,
            critic_desc=critic_desc,
            n_repeat=1
        )
        agents.append(f"Reflexion CoT agent {results_reflexion['cot_agent'].id}, refining reasoning and answer, thinking: {results_reflexion['list_thinking'][0].content}; answer: {results_reflexion['list_answer'][0].content}")
        for i_feedback in range(len(results_reflexion['list_feedback'])):
            agents.append(f"Critic agent {results_reflexion['critic_agent'].id}, feedback: {results_reflexion['list_feedback'][i_feedback].content}; correct: {results_reflexion['list_correct'][i_feedback].content}")
        sub_tasks.append(f"Subtask 2 output: thinking - {results_reflexion['list_thinking'][0].content}; answer - {results_reflexion['list_answer'][0].content}")
        logs.append(results_reflexion['subtask_desc'])
        improved_thinking_list.append(results_reflexion['list_thinking'][0])
        improved_answer_list.append(results_reflexion['list_answer'][0])
    # Control Flow 2: end_loop
    # Stage 2: Consolidate and select optimal output - subtask_3 (Aggregate)
    aggregate_instruction = "Subtask 3: Aggregate the improved reasoning outputs and select the most coherent and consistent final answer to the question of who did not join the alliance at Polin's request."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + improved_answer_list,
        'temperature': 0.0,
        'context': ["user query", "improved answers from iterative refinement"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, aggregating improved answers, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    # Stage 3: Validate Output (optional) - subtask_4 (Review | Reflexion)
    if hasattr(self, 'review') and hasattr(self, 'reflexion'):
        review_instruction = "Subtask 4: Validate the final answer against correctness and reliability criteria to ensure accuracy regarding who did not join the alliance at Polin's request."
        review_desc = {
            'instruction': review_instruction,
            'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
            'temperature': 0.0,
            'context': ["user query", "final thinking", "final answer"]
        }
        results_review = await self.review(
            subtask_id="subtask_4",
            review_desc=review_desc
        )
        agents.append(f"Review agent {results_review['review_agent'].id}, reviewing final answer, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
        sub_tasks.append(f"Subtask 4 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
        logs.append(results_review['subtask_desc'])
        # Optionally refine final answer based on review
        if results_review['correct'].content.lower() != 'yes':
            reflexion_instruction2 = "Subtask 4b: Refine the final answer based on review feedback to improve correctness and reliability."
            critic_instruction2 = "Please review the refined final answer and provide feedback."
            reflexion_desc2 = {
                'instruction': reflexion_instruction2,
                'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer'], results_review['feedback']],
                'output': ["thinking", "answer"],
                'temperature': 0.0,
                'context': ["user query", "final thinking", "final answer", "review feedback"]
            }
            critic_desc2 = {
                'instruction': critic_instruction2,
                'output': ["feedback", "correct"],
                'temperature': 0.0
            }
            results_refine = await self.reflexion(
                subtask_id="subtask_4b",
                cot_reflect_desc=reflexion_desc2,
                critic_desc=critic_desc2,
                n_repeat=1
            )
            agents.append(f"Reflexion CoT agent {results_refine['cot_agent'].id}, refining final answer after review, thinking: {results_refine['list_thinking'][0].content}; answer: {results_refine['list_answer'][0].content}")
            for i_fb in range(len(results_refine['list_feedback'])):
                agents.append(f"Critic agent {results_refine['critic_agent'].id}, feedback: {results_refine['list_feedback'][i_fb].content}; correct: {results_refine['list_correct'][i_fb].content}")
            sub_tasks.append(f"Subtask 4b output: thinking - {results_refine['list_thinking'][0].content}; answer - {results_refine['list_answer'][0].content}")
            logs.append(results_refine['subtask_desc'])
            final_thinking = results_refine['list_thinking'][0]
            final_answer = results_refine['list_answer'][0]
        else:
            final_thinking = results_aggregate['thinking']
            final_answer = results_aggregate['answer']
    else:
        final_thinking = results_aggregate['thinking']
        final_answer = results_aggregate['answer']
    # Control Flow 3: end_sequential
    final_output = await self.make_final_answer(final_thinking, final_answer, sub_tasks, agents)
    return final_output, logs