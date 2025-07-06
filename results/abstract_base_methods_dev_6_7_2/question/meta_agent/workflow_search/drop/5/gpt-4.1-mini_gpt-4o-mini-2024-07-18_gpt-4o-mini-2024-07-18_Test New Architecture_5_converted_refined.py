async def forward_5(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_reflect_instruction1 = "Subtask 1: Extract every occurrence of 'X-yard touchdown run' from the passage, list all touchdown run distances, and validate completeness by cross-checking the number of extracted runs against the number of 'touchdown run' mentions in the passage."
    critic_instruction1 = "Subtask 1: Review the extraction of touchdown runs, ensuring no runs are missed and the count matches the number of 'touchdown run' mentions. Provide feedback and corrections if needed."
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, extracting touchdown runs, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][k].content}; correct: {results1['list_correct'][k].content}")
        if k + 1 < len(results1['list_thinking']) and k + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining answer, thinking: {results1['list_thinking'][k+1].content}; answer: {results1['list_answer'][k+1].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    debate_instruction2 = "Subtask 2: Challenge the extraction by scanning the entire passage again for all 'touchdown run' occurrences, listing every distance found, and verifying no runs are missed."
    final_decision_instruction2 = "Subtask 2: Make final decision on the completeness and correctness of the touchdown run extraction."
    debate_desc2 = {
        "instruction": debate_instruction2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc2 = {
        "instruction": final_decision_instruction2,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        final_decision_desc=final_decision_desc2,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, challenging extraction, thinking: {results2['list_thinking'][round][idx].content}; answer: {results2['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, confirming extraction completeness, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    debate_instruction3 = "Subtask 3: List every phrase matching /\\d+-yard touchdown run/ in the passage and verify the completeness of the list by cross-checking with the total count of 'touchdown run' mentions."
    final_decision_instruction3 = "Subtask 3: Make final decision on the completeness and accuracy of the touchdown run list."
    debate_desc3 = {
        "instruction": debate_instruction3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc3 = {
        "instruction": final_decision_instruction3,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results3 = await self.debate(
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, verifying list completeness, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, confirming list accuracy, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    cot_sc_instruction4 = "Subtask 4: Perform at least three independent decompositions of the passage to identify touchdown run distances, then vote on the consolidated list to ensure no runs are missed and distances are accurate."
    N = max(3, self.max_sc)
    cot_sc_desc4 = {
        'instruction': cot_sc_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
    }
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_desc=cot_sc_desc4,
        n_repeat=N
    )
    for idx in range(N):
        agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, decomposition #{idx+1}, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    cot_reflect_instruction5 = "Subtask 5: Review the consolidated touchdown run distances and calculated yard difference, cross-verify that the number of extracted runs matches the number of 'touchdown run' mentions, and provide feedback on completeness and correctness."
    critic_instruction5 = "Subtask 5: Critically evaluate the review for any gaps or missing runs, and confirm the correctness of the yard difference calculation."
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"]
    }
    critic_desc5 = {
        'instruction': critic_instruction5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc5,
        critic_desc=critic_desc5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, reviewing consolidated runs, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, feedback: {results5['list_feedback'][i].content}; correct: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining answer, thinking: {results5['list_thinking'][i+1].content}; answer: {results5['list_answer'][i+1].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    cot_instruction6 = "Subtask 6: Final step-by-step chain-of-thought verification to confirm that all touchdown runs have been captured, and to calculate the minimum, maximum, and the difference in yards between the longest and shortest touchdown runs using the exhaustive list."
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, final verification and calculation, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
