async def forward_7(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_bands = []
    # Control Flow 0: start sequential
    # Stage 0: generate candidate outputs with CoT and Self-Consistency in a loop
    cot_instruction = "Sub-task 1: Generate candidate Lyric Street Records-affiliated bands that could have covered 'If You Ever Get Lonely' by applying structured reasoning and knowledge of the song's cover versions"
    for i in range(self.max_sc):
        cot_sc_desc = {
            'instruction': cot_instruction,
            'input': [taskInfo],
            'temperature': 0.5,
            'context': ["user query"]
        }
        results = await self.sc_cot(
            subtask_id=f"subtask_1_{i+1}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=1
        )
        agents.append(f"CoT-SC agent {results['cot_agent'][0].id}, iteration {i+1}, generating candidate bands, thinking: {results['list_thinking'][0]}; answer: {results['list_answer'][0]}")
        sub_tasks.append(f"Sub-task 1 iteration {i+1} output: thinking - {results['list_thinking'][0]}; answer - {results['list_answer'][0]}")
        logs.append(results['subtask_desc'])
        candidate_bands.append(results['list_answer'][0])
    # Control Flow 2: end loop
    # Stage 1: consolidate multiple inputs with Aggregate agent
    aggregate_instruction = "Sub-task 2: From candidate bands generated, aggregate these solutions and return the consistent and best solution identifying the Lyric Street Records-affiliated band that covered 'If You Ever Get Lonely'"
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_bands,
        'temperature': 0.0,
        'context': ["user query", "candidate bands generated from subtask 1"]
    }
    results_agg = await self.aggregate(
        subtask_id="subtask_2",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_agg['aggregate_agent'].id}, consolidating candidate bands, thinking: {results_agg['thinking'].content}; answer: {results_agg['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results_agg['thinking'].content}; answer - {results_agg['answer'].content}")
    logs.append(results_agg['subtask_desc'])
    # Stage 1: validate consolidated output with Review, Programmer, and CoT agents
    review_instruction = "Sub-task 3: Review the consolidated answer to ensure it accurately and completely identifies the correct Lyric Street Records-affiliated band that covered 'If You Ever Get Lonely'"
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
    sub_tasks.append(f"Sub-task 3 output: feedback - {results_review['thinking'].content}; correct - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    if results_review['answer'].content.lower() != 'correct':
        programmer_instruction = "Sub-task 4: Based on review feedback, refine and validate the consolidated answer identifying the Lyric Street Records-affiliated band that covered 'If You Ever Get Lonely'"
        programmer_desc = {
            'instruction': programmer_instruction,
            'input': [taskInfo, results_agg['thinking'], results_agg['answer'], results_review['thinking'], results_review['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "feedback of subtask 3", "correctness of subtask 3"]
        }
        results_prog = await self.programmer(
            subtask_id="subtask_4",
            programmer_desc=programmer_desc
        )
        agents.append(f"Programmer agent {results_prog['programmer_agent'].id}, refining consolidated answer, thinking: {results_prog['thinking'].content}; answer: {results_prog['answer'].content}, executing results: {results_prog['exec_result']}")
        sub_tasks.append(f"Sub-task 4 output: thinking - {results_prog['thinking'].content}; answer - {results_prog['answer'].content}; output - {results_prog['exec_result']}")
        logs.append(results_prog['subtask_desc'])
        cot_instruction_final = "Sub-task 5: Final reasoning to confirm the refined consolidated answer identifying the Lyric Street Records-affiliated band that covered 'If You Ever Get Lonely'"
        cot_agent_desc_final = {
            'instruction': cot_instruction_final,
            'input': [taskInfo, results_prog['thinking'], results_prog['answer']],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
        }
        results_final = await self.cot(
            subtask_id="subtask_5",
            cot_agent_desc=cot_agent_desc_final
        )
        agents.append(f"CoT agent {results_final['cot_agent'].id}, final reasoning, thinking: {results_final['thinking'].content}; answer: {results_final['answer'].content}")
        sub_tasks.append(f"Sub-task 5 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
        logs.append(results_final['subtask_desc'])
    else:
        results_final = results_agg
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs
