async def forward_9(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_answers = []
    # Control Flow 0: start sequential
    # Control Flow 1: start loop for candidate generation
    for i in range(self.max_sc):
        cot_instruction = f"Subtask 1: Systematically generate candidate answer #{i+1} identifying the Shakespeare tragedy with the fictional character Benvolio and the protagonist who secretly loves and marries a member of the rival house."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results = await self.sc_cot(
            subtask_id=f"subtask_{i+1}",
            cot_sc_desc=cot_agent_desc,
            n_repeat=1
        )
        candidate_answers.append(results['answer'].content)
        agents.append(f"CoT-SC agent {results['cot_agent'][0].id}, generating candidate #{i+1}, thinking: {results['list_thinking'][0]}; answer: {results['list_answer'][0]}")
        sub_tasks.append(f"Subtask {i+1} output: thinking - {results['list_thinking'][0]}; answer - {results['list_answer'][0]}")
        logs.append(results['subtask_desc'])
    # Control Flow 2: end loop
    # Stage 1: consolidate multiple inputs
    aggregate_instruction = "Subtask 2: From the candidate answers generated, aggregate these solutions and return the consistent and best solution identifying the protagonist character."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_answers,
        'temperature': 0.0,
        'context': ["user query", "candidate answers"]
    }
    results_agg = await self.aggregate(
        subtask_id="subtask_2",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_agg['aggregate_agent'].id}, integrating candidate answers, thinking: {results_agg['thinking'].content}; answer: {results_agg['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results_agg['thinking'].content}; answer - {results_agg['answer'].content}")
    logs.append(results_agg['subtask_desc'])
    # Stage 2: validate consolidated output
    review_instruction = "Subtask 3: Review the consolidated protagonist identification for accuracy, completeness, validity, and correctness."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_agg['thinking'], results_agg['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results_review = await self.review(
        subtask_id="subtask_3",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing consolidated answer, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    programmer_instruction = "Subtask 4: Generate Python code to verify the correctness of the consolidated protagonist identification."
    programmer_desc = {
        'instruction': programmer_instruction,
        'input': [taskInfo, results_agg['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 2"]
    }
    results_prog = await self.programmer(
        subtask_id="subtask_4",
        programmer_desc=programmer_desc
    )
    agents.append(f"Programmer agent {results_prog['programmer_agent'].id}, generating verification code, thinking: {results_prog['thinking'].content}; answer: {results_prog['answer'].content}; exec_result: {results_prog['exec_result']}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_prog['thinking'].content}; answer - {results_prog['answer'].content}; output - {results_prog['exec_result']}")
    logs.append(results_prog['subtask_desc'])
    cot_instruction_final = "Subtask 5: Based on review and verification, finalize the identification of the protagonist character in the Shakespeare tragedy with Benvolio and the secret marriage to a rival house member."
    cot_agent_desc_final = {
        'instruction': cot_instruction_final,
        'input': [taskInfo, results_agg['answer'], results_review['answer'], results_prog['answer']],
        'temperature': 0.0,
        'context': ["user query", "aggregated answer", "review feedback", "verification result"]
    }
    results_final = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc_final
    )
    agents.append(f"CoT agent {results_final['cot_agent'].id}, finalizing answer, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs