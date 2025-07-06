async def forward_12(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_outputs = []
    # Control Flow 0: start sequential
    # Stage 0: generate candidate outputs with CoT and SC-CoT in a loop
    # Control Flow 1: start loop
    for i in range(3):
        cot_instruction = f"Sub-task {i*2+1}: Calculate the handbag cost using the formula: handbag cost = 3 times shoe cost minus 20, considering possible interpretations iteration {i+1}."
        cot_agent_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results_cot = await self.cot(
            subtask_id=f"subtask_{i*2+1}",
            cot_agent_desc=cot_agent_desc
        )
        agents.append(f"CoT agent {results_cot['cot_agent'].id}, iteration {i+1}, thinking: {results_cot['thinking'].content}; answer: {results_cot['answer'].content}")
        sub_tasks.append(f"Sub-task {i*2+1} output: thinking - {results_cot['thinking'].content}; answer - {results_cot['answer'].content}")
        logs.append(results_cot['subtask_desc'])
        cot_sc_instruction = f"Sub-task {i*2+2}: Refine the calculation of handbag cost based on previous CoT output with self-consistency, iteration {i+1}."
        cot_sc_desc = {
            'instruction': cot_sc_instruction,
            'input': [taskInfo, results_cot['thinking'], results_cot['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of previous CoT", "answer of previous CoT"]
        }
        results_sc = await self.sc_cot(
            subtask_id=f"subtask_{i*2+2}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=self.max_sc
        )
        for idx in range(self.max_sc):
            agents.append(f"CoT-SC agent {results_sc['cot_agent'][idx].id}, iteration {i+1}, consider cases, thinking: {results_sc['list_thinking'][idx]}; answer: {results_sc['list_answer'][idx]}")
        sub_tasks.append(f"Sub-task {i*2+2} output: thinking - {results_sc['thinking'].content}; answer - {results_sc['answer'].content}")
        logs.append(results_sc['subtask_desc'])
        candidate_outputs.append(results_sc['answer'].content)
    # Control Flow 2: end loop
    # Stage 1: consolidate multiple inputs
    aggregate_instruction = "Sub-task 7: From candidate handbag cost outputs, aggregate these solutions and return the consistent and best handbag cost value."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_outputs,
        'temperature': 0.0,
        'context': ["user query", "candidate outputs"]
    }
    results_aggregate = await self.aggregate(
        subtask_id="subtask_7",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_aggregate['aggregate_agent'].id}, thinking: {results_aggregate['thinking'].content}; answer: {results_aggregate['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results_aggregate['thinking'].content}; answer - {results_aggregate['answer'].content}")
    logs.append(results_aggregate['subtask_desc'])
    # Stage 1: validate consolidated output
    review_instruction = "Sub-task 8: Review the consolidated handbag cost output for accuracy and correctness."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_aggregate['thinking'], results_aggregate['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of aggregation", "answer of aggregation"]
    }
    results_review = await self.review(
        subtask_id="subtask_8",
        review_desc=review_desc
    )
    agents.append(f"Review agent {results_review['review_agent'].id}, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Sub-task 8 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    # Control Flow 3: end sequential
    final_answer = await self.make_final_answer(results_review['thinking'], results_review['answer'], sub_tasks, agents)
    return final_answer, logs