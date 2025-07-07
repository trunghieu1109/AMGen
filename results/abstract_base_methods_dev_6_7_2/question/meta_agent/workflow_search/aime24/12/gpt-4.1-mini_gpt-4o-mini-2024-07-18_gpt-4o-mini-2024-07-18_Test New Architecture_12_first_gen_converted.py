async def forward_12(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Analyze the expression (75+117i)z + (96+144i)/z with the constraint |z|=4, formulate and verify formal relationships among variables to determine parameter values that maximize the real part."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing expression and constraints, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    candidates = []
    for i in range(self.max_sc):
        answer_generate_instruction = f"Sub-task 2.{i+1}: Generate candidate values of z and compute the real part of (75+117i)z + (96+144i)/z under |z|=4, documenting reasoning."
        answer_generate_desc = {
            'instruction': answer_generate_instruction,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
        }
        results2 = await self.answer_generate(
            subtask_id=f"subtask_2_{i+1}",
            cot_agent_desc=answer_generate_desc
        )
        agents.append(f"AnswerGenerate agent {results2['cot_agent'].id}, generating candidate {i+1}, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
        sub_tasks.append(f"Sub-task 2.{i+1} output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
        logs.append(results2['subtask_desc'])
        candidates.append((results2['thinking'], results2['answer']))

    aggregate_instruction3 = "Sub-task 3: Aggregate candidate solutions from Sub-task 2, evaluate their consistency, select the candidate with the largest real part, transform into the specified output format, and validate accuracy and completeness."
    aggregate_desc = {
        'instruction': aggregate_instruction3,
        'input': [taskInfo] + candidates,
        'temperature': 0.0,
        'context': ["user query", "candidate solutions from subtask 2"]
    }
    results3 = await self.aggregate(
        subtask_id="subtask_3",
        aggregate_desc=aggregate_desc
    )
    agents.append(f"Aggregate agent {results3['aggregate_agent'].id}, aggregating candidates, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    reflexion_instruction4 = "Sub-task 4: Based on aggregation results, review and reflect to confirm the largest possible real part and ensure no inconsistencies remain."
    critic_instruction4 = "Please review the aggregated solution for correctness and completeness."
    cot_reflect_desc4 = {
        'instruction': reflexion_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id="subtask_4",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, reflecting on aggregated solution, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback: {results4['list_feedback'][i].content}; correct: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining final answer, thinking: {results4['list_thinking'][i+1].content}; answer: {results4['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
