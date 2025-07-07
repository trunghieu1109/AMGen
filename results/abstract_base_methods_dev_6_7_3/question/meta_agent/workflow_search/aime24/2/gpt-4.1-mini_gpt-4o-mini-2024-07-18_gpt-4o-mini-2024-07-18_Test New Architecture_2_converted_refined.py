async def forward_2(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Subtask 1: Formalize the problem of coloring vertices of a regular octagon with red or blue independently with equal probability. Define variables representing colorings as subsets of vertices, formalize the rotation group action, and clarify that the problem requires existence of a rotation mapping blue vertices to original red vertices, not invariance under all rotations."
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, formalizing problem, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction2 = "Subtask 2: Using the formalization from Subtask 1, enumerate fixed points of colorings under each rotation of the octagon. For each rotation, calculate the number of colorings fixed by that rotation using Burnside's lemma. Consider multiple cases with self-consistency to explore different rotation effects and fixed points."
    N = self.max_sc
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=N
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    for idx in range(len(results2['list_thinking'])):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, enumerating fixed points under rotations, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = "Subtask 3: Critically evaluate and verify the combinatorial calculations and assumptions from Subtask 2. Recalculate or correct the number of valid colorings fixed by rotations if necessary, ensuring the application of Burnside's lemma is rigorous and complete."
    critic_instruction3 = "Please review the combinatorial counting and provide feedback on limitations or errors, and suggest corrections."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    critic_desc3 = {
        'instruction': critic_instruction3,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results3 = await self.reflexion(
        subtask_id="subtask_3",
        cot_reflect_desc=cot_reflect_desc3,
        critic_desc=critic_desc3,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, verifying combinatorial counts, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correction: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining answer, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    candidates = []
    for rotation_idx in range(8):
        cot_instruction4 = f"Subtask 4: For rotation {rotation_idx}, explicitly calculate the number of colorings fixed by this rotation using Burnside's lemma. Provide detailed reasoning on how this rotation affects the coloring and count fixed points."
        cot_agent_desc4 = {
            'instruction': cot_instruction4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.0,
            'context': ["user query", "verified combinatorial counts"]
        }
        results4 = await self.cot(
            subtask_id=f"subtask_4_{rotation_idx + 1}",
            cot_agent_desc=cot_agent_desc4
        )
        agents.append(f"CoT agent {results4['cot_agent'].id}, detailed fixed point calculation for rotation {rotation_idx}, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
        sub_tasks.append(f"Sub-task 4_{rotation_idx + 1} output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])
        candidates.append(results4)

    debate_instruction5 = "Subtask 5: Based on the candidate solutions from Subtask 4, debate the correctness and consistency of each candidate's reasoning and probability contribution. Agents should argue for or against candidates to select the most coherent and justified probability value."
    final_decision_instruction5 = "Subtask 5: Make final decision on the aggregated probability value after debate."
    debate_desc5 = {
        'instruction': debate_instruction5,
        'context': ["user query"] + [c['thinking'] for c in candidates] + [c['answer'] for c in candidates],
        'input': [taskInfo] + [c['answer'] for c in candidates],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc5 = {
        'instruction': final_decision_instruction5,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc5,
        final_decision_desc=final_decision_desc5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating candidate solutions, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding final probability, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_instruction6 = "Subtask 6: Validate the final fraction m/n representing the probability. Confirm that m and n are relatively prime positive integers, and output the integer m+n as the final answer."
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "final debated probability"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, validating fraction and formatting final answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_reflect_instruction7 = "Subtask 7: Reflect on the entire solution process, identify any remaining errors or inconsistencies, and provide a corrected, clear, and justified final solution."
    critic_instruction7 = "Please review the final solution and provide feedback or corrections if needed."
    cot_reflect_desc7 = {
        'instruction': cot_reflect_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "validated final answer"]
    }
    critic_desc7 = {
        'instruction': critic_instruction7,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results7 = await self.reflexion(
        subtask_id="subtask_7",
        cot_reflect_desc=cot_reflect_desc7,
        critic_desc=critic_desc7,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, reflecting on final solution, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, feedback: {results7['list_feedback'][i].content}; correction: {results7['list_correct'][i].content}")
        if i + 1 < len(results7['list_thinking']) and i + 1 < len(results7['list_answer']):
            agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, refining final solution, thinking: {results7['list_thinking'][i + 1].content}; answer: {results7['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_instruction9 = "Subtask 9: Provide a concise, stepwise final answer to the problem, outputting only the integer m+n without extraneous explanation."
    cot_agent_desc9 = {
        'instruction': cot_instruction9,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'temperature': 0.0,
        'context': ["user query", "refined final solution"]
    }
    results9 = await self.cot(
        subtask_id="subtask_9",
        cot_agent_desc=cot_agent_desc9
    )
    agents.append(f"CoT agent {results9['cot_agent'].id}, generating final concise answer, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
