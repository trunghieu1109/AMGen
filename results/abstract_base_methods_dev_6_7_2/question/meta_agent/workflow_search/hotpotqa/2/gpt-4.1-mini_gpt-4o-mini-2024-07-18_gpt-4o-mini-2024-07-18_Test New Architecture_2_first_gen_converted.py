async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    # Control Flow 0: start sequential
    
    # Control Flow 1: start loop
    candidate_outputs = []
    for i in range(self.max_sc):
        cot_sc_instruction = f"Subtask 1: Generate candidate governor after the Missouri Compromise by applying structured reasoning, attempt {i+1}"
        cot_sc_desc = {
            'instruction': cot_sc_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results_sc = await self.sc_cot(
            subtask_id=f"subtask_1_{i+1}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results_sc['cot_agent'][0].id}, attempt {i+1}, thinking: {results_sc['list_thinking'][0]}; answer: {results_sc['list_answer'][0]}")
        sub_tasks.append(f"Subtask 1 attempt {i+1} output: thinking - {results_sc['list_thinking'][0]}; answer - {results_sc['list_answer'][0]}")
        logs.append(results_sc['subtask_desc'])
        candidate_outputs.append(results_sc['list_answer'][0])
    
    # Control Flow 2: end loop
    
    # Stage 1: consolidate multiple inputs
    aggregate_instruction = "Subtask 2: From candidate governors generated, aggregate these solutions and return the consistent and best identification of the first governor after the Missouri Compromise"
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_outputs,
        'temperature': 0.0,
        'context': ["user query", "candidate governors from subtask 1"]
    }
    results_agg = await self.aggregate(
        subtask_id="subtask_2",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_agg['aggregate_agent'].id}, consolidating candidate governors, thinking: {results_agg['thinking'].content}; answer: {results_agg['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results_agg['thinking'].content}; answer - {results_agg['answer'].content}")
    logs.append(results_agg['subtask_desc'])
    
    # Stage 1: validate consolidated output
    review_instruction = "Subtask 3: Review the consolidated identification of the first governor after the Missouri Compromise to confirm accuracy and completeness, then determine the governor's place of origin"
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
    agents.append(f"Review agent {results_review['review_agent'].id}, reviewing consolidated governor identification, feedback: {results_review['thinking'].content}; correct: {results_review['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    
    programmer_instruction = "Subtask 4: Generate Python code to verify and extract the place of origin of the identified governor"
    programmer_desc = {
        'instruction': programmer_instruction,
        'input': [taskInfo, results_agg['answer'], results_review['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 2", "answer of subtask 3"],
        'entry_point': "verify_and_extract_origin"
    }
    results_prog = await self.programmer(
        subtask_id="subtask_4",
        programmer_desc=programmer_desc
    )
    agents.append(f"Programmer agent {results_prog['programmer_agent'].id}, generating verification code, thinking: {results_prog['thinking'].content}; answer: {results_prog['answer'].content}; output: {results_prog['exec_result']}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_prog['thinking'].content}; answer - {results_prog['answer'].content}; output - {results_prog['exec_result']}")
    logs.append(results_prog['subtask_desc'])
    
    cot_instruction_final = "Subtask 5: Based on the verified output, provide a final answer stating the place of origin of the first governor after the Missouri Compromise"
    cot_desc_final = {
        'instruction': cot_instruction_final,
        'input': [taskInfo, results_prog['thinking'], results_prog['exec_result']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "output of subtask 4"]
    }
    results_final = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_desc_final
    )
    agents.append(f"CoT agent {results_final['cot_agent'].id}, finalizing answer, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs