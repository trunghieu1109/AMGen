async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_instruction0 = "Subtask 1: Categorize the problem type and verify given properties: triangle ABC with sides AB=5, BC=9, AC=10, circle ω, tangents at B and C, intersection D, and line AD intersecting ω at P"
    cot_agent_desc0 = {
        'instruction': cot_instruction0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0 = await self.cot(
        subtask_id="subtask_1",
        cot_agent_desc=cot_agent_desc0
    )
    agents.append(f"CoT agent {results0['cot_agent'].id}, categorizing problem and verifying properties, thinking: {results0['thinking'].content}; answer: {results0['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])
    
    cot_instruction1 = "Subtask 2: Extract and list essential geometric features and relationships from the problem statement, including circle ω, tangents at B and C, point D, and segment AP"
    cot_agent_desc1 = {
        'instruction': cot_instruction1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results1 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc1
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting essential features, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_sc_instruction2 = "Subtask 3: Generate multiple candidate solution approaches independently to find AP, including power of a point and symmedian methods. Each approach must explicitly verify each step with geometric theorems and numeric checks before finalizing answers."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, candidate solution approach {idx+1}, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    debate_instruction3 = "Subtask 4: Debate among candidate solutions from Subtask 3 to resolve conflicts and select the most coherent and validated method. Agents must provide detailed justifications and counterarguments for their candidate answers."
    final_decision_instruction3 = "Subtask 4: Make final decision on the most valid candidate solution for AP length."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"],
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc3 = {
        'instruction': final_decision_instruction3,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results3 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round+1}, debating candidate solutions, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding best solution, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    cot_reflect_instruction4 = "Subtask 5: Review and refine the selected solution from Subtask 4, focusing on clarity, correctness, and stepwise derivation of AP using power of a point or symmedian properties. Verify all formulas and simplify expressions before concluding."
    critic_instruction4 = "Please review the refined solution for AP length, identify any limitations or errors, and suggest corrections if needed."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc4 = {
        'instruction': critic_instruction4,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc4,
        critic_desc=critic_desc4,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining solution, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback round {i+1}, thinking: {results4['list_feedback'][i].content}; answer: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining further, thinking: {results4['list_thinking'][i+1].content}; answer: {results4['list_answer'][i+1].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    cot_instruction5 = "Subtask 6: Format the finalized AP length solution into the required concise numeric answer format: output m+n where AP = m/n in lowest terms"
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "refined solution"]
    }
    results5 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc5
    )
    agents.append(f"CoT agent {results5['cot_agent'].id}, formatting final numeric answer, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
