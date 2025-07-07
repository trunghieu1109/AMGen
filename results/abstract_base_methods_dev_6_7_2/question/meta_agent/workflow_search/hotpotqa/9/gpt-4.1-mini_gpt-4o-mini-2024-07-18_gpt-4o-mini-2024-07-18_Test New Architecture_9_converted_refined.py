async def forward_9(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    candidate_answers = []
    cot_reflect_instruction1 = "Sub-task 1: Verify the factual correctness of the premise that Benvolio slays a character in the Shakespeare tragedy and identify any inconsistencies or errors before proceeding to identify the protagonist who secretly loves and marries a member of the rival house."
    critic_instruction1 = "Please review the factual accuracy of the premise about Benvolio's actions and provide any corrections or clarifications needed."
    cot_reflect_desc1 = {
        'instruction': cot_reflect_instruction1,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc1 = {
        'instruction': critic_instruction1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc1,
        critic_desc=critic_desc1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, verifying premise about Benvolio, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, providing feedback, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining premise verification, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    for i in range(self.max_sc):
        cot_sc_instruction2 = f"Sub-task 2: Generate candidate answer #{i+1} identifying the protagonist in the Shakespeare tragedy involving Benvolio and the secret marriage to a rival house member. Provide detailed justifications and highlight any uncertainties or ambiguities in the question."
        cot_sc_desc = {
            'instruction': cot_sc_instruction2,
            'input': [taskInfo, results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "premise verification"]
        }
        results2 = await self.sc_cot(
            subtask_id=f"subtask_{i+2}",
            cot_sc_desc=cot_sc_desc,
            n_repeat=1
        )
        candidate_answers.append(results2['answer'].content)
        agents.append(f"CoT-SC agent {results2['cot_agent'][0].id}, generating candidate #{i+1}, thinking: {results2['list_thinking'][0]}; answer: {results2['list_answer'][0]}")
        sub_tasks.append(f"Sub-task 2 output #{i+1}: thinking - {results2['list_thinking'][0]}; answer - {results2['list_answer'][0]}")
        logs.append(results2['subtask_desc'])
    aggregate_instruction = "Sub-task 3: Aggregate the candidate answers generated to produce a consistent and best solution identifying the protagonist character, cross-checking for factual accuracy especially regarding Benvolio's actions."
    aggregate_desc = {
        'instruction': aggregate_instruction,
        'input': [taskInfo] + candidate_answers + [results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "candidate answers", "premise verification"]
    }
    results_agg = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results_agg['aggregate_agent'].id}, integrating candidate answers, thinking: {results_agg['thinking'].content}; answer: {results_agg['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results_agg['thinking'].content}; answer - {results_agg['answer'].content}")
    logs.append(results_agg['subtask_desc'])
    review_instruction = "Sub-task 4: Review the consolidated protagonist identification for accuracy, completeness, and especially verify the factual correctness of Benvolio's actions in the play."
    review_desc = {
        'instruction': review_instruction,
        'input': [taskInfo, results_agg['thinking'], results_agg['answer'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of sub-task 3", "answer of sub-task 3", "premise verification"]
    }
    results_review = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc={
            'instruction': review_instruction,
            'input': [taskInfo, results_agg['thinking'], results_agg['answer'], results1['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of sub-task 3", "answer of sub-task 3", "premise verification"]
        },
        critic_desc={
            'instruction': "Please critically evaluate the consolidated answer and verify the factual accuracy of Benvolio's actions.",
            'output': ["feedback", "correct"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_review['cot_agent'].id}, reviewing consolidated answer, thinking: {results_review['list_thinking'][0].content}; answer: {results_review['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results_review['list_feedback']))):
        agents.append(f"Critic agent {results_review['critic_agent'].id}, providing feedback, thinking: {results_review['list_feedback'][i].content}; answer: {results_review['list_correct'][i].content}")
        if i + 1 < len(results_review['list_thinking']) and i + 1 < len(results_review['list_answer']):
            agents.append(f"Reflexion CoT agent {results_review['cot_agent'].id}, refining review, thinking: {results_review['list_thinking'][i + 1].content}; answer: {results_review['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results_review['thinking'].content}; answer - {results_review['answer'].content}")
    logs.append(results_review['subtask_desc'])
    programmer_instruction = "Sub-task 5: Generate Python code to verify the correctness of the consolidated protagonist identification and the factual accuracy of Benvolio's actions in the play."
    programmer_desc = {
        'instruction': programmer_instruction,
        'input': [taskInfo, results_agg['answer'], results_review['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of sub-task 3", "review feedback"]
    }
    results_prog = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=programmer_desc
    )
    agents.append(f"CoT agent {results_prog['cot_agent'].id}, generating verification code, thinking: {results_prog['thinking'].content}; answer: {results_prog['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results_prog['thinking'].content}; answer - {results_prog['answer'].content}")
    logs.append(results_prog['subtask_desc'])
    cot_reflect_instruction_final = "Sub-task 6: Based on all previous outputs, finalize the identification of the protagonist character, explicitly clarifying that Benvolio does not slay any character in the play and addressing any misleading or incorrect premises in the original question."
    critic_instruction_final = "Please review the final answer for completeness, factual accuracy, and clarity regarding the misleading premise."
    cot_reflect_desc_final = {
        'instruction': cot_reflect_instruction_final,
        'input': [taskInfo, results_agg['answer'], results_review['answer'], results_prog['answer'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "aggregated answer", "review feedback", "verification result", "premise verification"]
    }
    critic_desc_final = {
        'instruction': critic_instruction_final,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results_final = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc_final,
        critic_desc=critic_desc_final,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results_final['cot_agent'].id}, finalizing answer, thinking: {results_final['list_thinking'][0].content}; answer: {results_final['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results_final['list_feedback']))):
        agents.append(f"Critic agent {results_final['critic_agent'].id}, providing feedback, thinking: {results_final['list_feedback'][i].content}; answer: {results_final['list_correct'][i].content}")
        if i + 1 < len(results_final['list_thinking']) and i + 1 < len(results_final['list_answer']):
            agents.append(f"Reflexion CoT agent {results_final['cot_agent'].id}, refining final answer, thinking: {results_final['list_thinking'][i + 1].content}; answer: {results_final['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results_final['thinking'].content}; answer - {results_final['answer'].content}")
    logs.append(results_final['subtask_desc'])
    final_answer = await self.make_final_answer(results_final['thinking'], results_final['answer'], sub_tasks, agents)
    return final_answer, logs
