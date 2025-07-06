async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_outputs = []
    for i in range(self.max_sc):
        cot_sc_instruction = (
            "Subtask 1: Generate candidate for the first elected governor of Missouri after statehood in 1820, "
            "explicitly verifying with official historical records and clarifying the definition of 'first governor after the Missouri Compromise'. "
            "Provide detailed reasoning and cite sources if possible."
        )
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
    aggregate_instruction = (
        "Subtask 2: Aggregate the candidate governors generated in Subtask 1, "
        "evaluate their consistency and reliability, and return the best supported identification of the first elected governor of Missouri after statehood in 1820."
    )
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_outputs,
        'temperature': 0.0,
        'context': ["user query", "candidate governors from subtask 1"]
    }
    results_agg = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=aggregate_desc,
        n_repeat=self.max_sc
    )
    agents.append(f"CoT-SC agent {results_agg['cot_agent'][0].id}, aggregating candidate governors, thinking: {results_agg['list_thinking'][0]}; answer: {results_agg['list_answer'][0]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results_agg['list_thinking'][0]}; answer - {results_agg['list_answer'][0]}")
    logs.append(results_agg['subtask_desc'])
    cot_reflect_instruction = (
        "Subtask 3: Review the aggregated identification of the first governor after the Missouri Compromise, "
        "explicitly define and justify the interpretation of 'from' in the query (e.g., birthplace, political base, or state of governance). "
        "Then determine the place of origin accordingly, providing reasoning and clarifications."
    )
    critic_instruction = (
        "Please review the interpretation and justification of 'from' and the determination of the governor's place of origin, "
        "pointing out any limitations or ambiguities."
    )
    cot_reflect_desc = {
        'instruction': cot_reflect_instruction,
        'input': [taskInfo, results_agg['list_thinking'][0], results_agg['list_answer'][0]],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc = {
        'instruction': critic_instruction,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results_reflect = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc,
        critic_desc=critic_desc,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_reflect['cot_agent'].id}, clarifying interpretation of 'from', thinking: {results_reflect['list_thinking'][0].content}; answer: {results_reflect['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results_reflect['list_feedback']))):
        agents.append(f"Critic agent {results_reflect['critic_agent'].id}, providing feedback, thinking: {results_reflect['list_feedback'][i].content}; answer: {results_reflect['list_correct'][i].content}")
        if i + 1 < len(results_reflect['list_thinking']) and i + 1 < len(results_reflect['list_answer']):
            agents.append(f"Reflexion CoT agent {results_reflect['cot_agent'].id}, refining answer, thinking: {results_reflect['list_thinking'][i + 1].content}; answer: {results_reflect['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results_reflect['thinking'].content}; answer - {results_reflect['answer'].content}")
    logs.append(results_reflect['subtask_desc'])
    cot_instruction_code = (
        "Subtask 4: Generate Python code to verify and extract the place of origin of the identified governor, "
        "including comments clarifying data sources and assumptions used in the verification process."
    )
    cot_code_desc = {
        'instruction': cot_instruction_code,
        'input': [taskInfo, results_agg['list_answer'][0], results_reflect['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 2", "answer of subtask 3"],
        'entry_point': "verify_and_extract_origin"
    }
    results_code = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_code_desc
    )
    agents.append(f"CoT agent {results_code['cot_agent'].id}, generating verification code, thinking: {results_code['thinking'].content}; answer: {results_code['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results_code['thinking'].content}; answer - {results_code['answer'].content}")
    logs.append(results_code['subtask_desc'])
    cot_reflect_instruction_final = (
        "Subtask 5: Based on all previous outputs, provide a final answer stating the place of origin of the first governor after the Missouri Compromise. "
        "Explicitly state and justify the interpretation of 'from' used, and if ambiguity remains, provide both birthplace and political origin with explanation."
    )
    critic_instruction_final = (
        "Please review the final answer for completeness, clarity, and correctness, especially regarding the interpretation of 'from'."
    )
    cot_reflect_desc_final = {
        'instruction': cot_reflect_instruction_final,
        'input': [taskInfo, results_agg['list_answer'][0], results_reflect['answer'], results_code['thinking'], results_code['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 2", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc_final = {
        'instruction': critic_instruction_final,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results_final = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc_final,
        critic_desc=critic_desc_final,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_final['cot_agent'].id}, finalizing answer, thinking: {results_final['list_thinking'][0].content}; answer: {results_final['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results_final['list_feedback']))):
        agents.append(f"Critic agent {results_final['critic_agent'].id}, providing feedback, thinking: {results_final['list_feedback'][i].content}; answer: {results_final['list_correct'][i].content}")
        if i + 1 < len(results_final['list_thinking']) and i + 1 < len(results_final['list_answer']):
            agents.append(f"Reflexion CoT agent {results_final['cot_agent'].id}, refining final answer, thinking: {results_final['list_thinking'][i + 1].content}; answer: {results_final['list_answer'][i + 1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs
