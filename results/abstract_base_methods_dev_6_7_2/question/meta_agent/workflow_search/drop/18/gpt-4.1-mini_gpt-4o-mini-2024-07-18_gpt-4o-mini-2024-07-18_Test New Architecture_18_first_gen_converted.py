async def forward_18(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Start sequential flow
    
    # Start loop flow for iterative quality enhancement
    loop_iterations = 3
    refined_interpretations = []
    for i in range(loop_iterations):
        reflexion_instruction = f"Subtask {i+1}: Iteratively evaluate and modify the initial interpretation of the passage to enhance clarity, consistency, and completeness regarding Matt Prater's field goals. Iteration {i+1}."
        critic_instruction = f"Please review the revised interpretation from iteration {i+1} and provide its limitations."
        reflexion_desc = {
            'instruction': reflexion_instruction,
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
        refined_interpretations.append(results_reflexion['answer'].content)
    
    # End loop flow
    
    # Stage 0: Decompose passage info into logical sequence to derive initial count (CoT + AnswerGenerate)
    cot_instruction = "Sub-task 4: Decompose the passage information into an ordered logical sequence to derive the initial count of field goals made by Matt Prater."
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
    agents.append(f"CoT agent {results_cot['cot_agent'].id}, decomposing passage, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
    logs.append(results_cot['subtask_desc'])
    
    answer_generate_instruction = "Sub-task 5: Generate an answer for the initial count of Matt Prater's field goals based on the logical sequence derived."
    answer_generate_desc = {
        'instruction': answer_generate_instruction,
        'input': [taskInfo, results_cot['thinking'], results_cot['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    results_answer_generate = await self.answer_generate(
        subtask_id="subtask_5",
        cot_agent_desc=answer_generate_desc
    )
    agents.append(f"AnswerGenerate agent {results_answer_generate['cot_agent'].id}, generating answer, thinking: {results_answer_generate['thinking'].content}; answer: {results_answer_generate['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results_answer_generate['thinking'].content}; answer - {results_answer_generate['answer'].content}")
    logs.append(results_answer_generate['subtask_desc'])
    
    # Stage 2: Aggregate multiple improved interpretations and select the most coherent final count (Aggregate)
    aggregate_instruction = "Sub-task 6: Aggregate multiple improved interpretations and select the most coherent and consistent final count of Matt Prater's field goals."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + refined_interpretations + [results_answer_generate['answer'].content],
        'temperature': 0.0,
        'context': ["user query", "improved interpretations", "initial count answer"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_6",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, aggregating interpretations, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    
    # Stage 3: Optionally evaluate the final answer for correctness and reliability (Review + Reflexion)
    review_instruction = "Sub-task 7: Evaluate the final answer for correctness and reliability against the passage details."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results_review = await self.review(
        subtask_id="subtask_7",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing final answer, feedback: {results_review['feedback'].content}; correct: {results_review['correct'].content}")
    sub_tasks.append(f"Sub-task 7 output: feedback - {results_review['feedback'].content}; correct - {results_review['correct'].content}")
    logs.append(results_review['subtask_desc'])
    
    reflexion_instruction_final = "Sub-task 8: Based on the review feedback, refine and finalize the answer for the number of field goals Matt Prater made."
    reflexion_desc_final = {
        'instruction': reflexion_instruction_final,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer'], results_review['feedback'], results_review['correct']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6", "feedback of subtask 7", "correctness of subtask 7"]
    }
    critic_desc_final = {
        'instruction': "Please review the final refined answer and provide any last feedback.",
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results_reflexion_final = await self.reflexion(
        subtask_id="subtask_8",
        cot_reflect_desc=reflexion_desc_final,
        critic_desc=critic_desc_final,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_reflexion_final['cot_agent'].id}, final refinement, thinking: {results_reflexion_final['list_thinking'][0].content}; answer: {results_reflexion_final['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results_reflexion_final['list_feedback']))):
        agents.append(f"Critic agent {results_reflexion_final['critic_agent'].id}, final feedback: {results_reflexion_final['list_feedback'][k].content}; correct: {results_reflexion_final['list_correct'][k].content}")
        if k + 1 < len(results_reflexion_final['list_thinking']) and k + 1 < len(results_reflexion_final['list_answer']):
            agents.append(f"Reflexion CoT agent {results_reflexion_final['cot_agent'].id}, refining final answer, thinking: {results_reflexion_final['list_thinking'][k + 1].content}; answer: {results_reflexion_final['list_answer'][k + 1].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results_reflexion_final['thinking'].content}; answer - {results_reflexion_final['answer'].content}")
    logs.append(results_reflexion_final['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_reflexion_final['thinking'], results_reflexion_final['answer'], sub_tasks, agents)
    return final_answer, logs
