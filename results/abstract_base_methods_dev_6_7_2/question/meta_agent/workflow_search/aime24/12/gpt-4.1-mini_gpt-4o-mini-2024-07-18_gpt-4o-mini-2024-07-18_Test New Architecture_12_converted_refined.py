async def forward_12(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = "Sub-task 1: Explicitly derive the real part of (75+117i)z + (96+144i)/z with |z|=4, showing detailed algebraic steps, differentiate carefully to find critical points, and verify these points numerically."
    critic_instruction1 = "Please review the algebraic derivation, differentiation, and verification steps for correctness and completeness."
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, deriving and verifying real part function, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining derivation, thinking: {results1['list_thinking'][i+1].content}; answer: {results1['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    candidates = []
    for i in range(self.max_sc):
        cot_instruction2_1 = f"Sub-task 2.{i+1}: Compute and document the real part of (75+117i)z + (96+144i)/z for candidate angle θ values (including 0, π/4, π/2, 3π/4, π, and critical points from Subtask 1), with |z|=4."
        cot_agent_desc2_1 = {
            'instruction': cot_instruction2_1,
            'input': [taskInfo, results1['thinking'], results1['answer']],
            'temperature': 0.5,
            'context': ["user query", "derivation and verification from subtask 1"]
        }
        results2_1 = await self.cot(
            subtask_id=f"subtask_2_1_{i+1}",
            cot_agent_desc=cot_agent_desc2_1
        )
        agents.append(f"CoT agent {results2_1['cot_agent'].id}, computing real parts for candidate θ values, thinking: {results2_1['thinking'].content}; answer: {results2_1['answer'].content}")
        sub_tasks.append(f"Sub-task 2.{i+1} output: thinking - {results2_1['thinking'].content}; answer - {results2_1['answer'].content}")
        logs.append(results2_1['subtask_desc'])
        candidates.append((results2_1['thinking'], results2_1['answer']))

    debate_instruction2_2 = "Sub-task 2.2: Debate among agents to verify candidate θ values and their computed real parts, challenge conclusions, and ensure the maximum real part is correctly identified."
    final_decision_instruction2_2 = "Sub-task 2.2: Make final decision on the maximum real part from debated candidates."
    debate_desc2_2 = {
        "instruction": debate_instruction2_2,
        "context": ["user query"] + [c[1] for c in candidates],
        "input": [taskInfo] + [c[0] for c in candidates] + [c[1] for c in candidates],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc2_2 = {
        "instruction": final_decision_instruction2_2,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results2_2 = await self.debate(
        subtask_id="subtask_2_2",
        debate_desc=debate_desc2_2,
        final_decision_desc=final_decision_desc2_2,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results2_2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, verifying candidate θ values and max real part, thinking: {results2_2['list_thinking'][round][idx].content}; answer: {results2_2['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding max real part, thinking: {results2_2['thinking'].content}; answer: {results2_2['answer'].content}")
    sub_tasks.append(f"Sub-task 2.2 output: thinking - {results2_2['thinking'].content}; answer - {results2_2['answer'].content}")
    logs.append(results2_2['subtask_desc'])

    cot_instruction2_3 = "Sub-task 2.3: Sample θ values densely around critical points and other candidates, compute and numerically evaluate the real parts of the expression to verify the maximum real part."
    cot_agent_desc2_3 = {
        'instruction': cot_instruction2_3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2_2['thinking'], results2_2['answer']],
        'temperature': 0.5,
        'context': ["user query", "derivation from subtask 1", "debate results from subtask 2.2"]
    }
    results2_3 = await self.cot(
        subtask_id="subtask_2_3",
        cot_agent_desc=cot_agent_desc2_3
    )
    agents.append(f"CoT agent {results2_3['cot_agent'].id}, densely sampling θ values and evaluating real parts, thinking: {results2_3['thinking'].content}; answer: {results2_3['answer'].content}")
    sub_tasks.append(f"Sub-task 2.3 output: thinking - {results2_3['thinking'].content}; answer - {results2_3['answer'].content}")
    logs.append(results2_3['subtask_desc'])

    reflexion_instruction4 = "Sub-task 4: Re-derive the maximum real part from first principles using the verified candidate θ values, numerically verify by substitution into the original expression, and confirm the correctness and completeness of the final maximum real part."
    critic_instruction4 = "Please review the re-derivation and numerical verification for correctness and completeness."
    cot_reflect_desc4 = {
        'instruction': reflexion_instruction4,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2_3['thinking'], results2_3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "derivation from subtask 1", "evaluation from subtask 2.3"]
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
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, final re-derivation and verification, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback: {results4['list_feedback'][i].content}; correct: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining final answer, thinking: {results4['list_thinking'][i+1].content}; answer: {results4['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
