async def forward_15(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_reflect_instruction1 = "Sub-task 1: Identify and explicitly verify candidate awards for which the expanded 2008 magazine article 'Is Google Making Us Stoopid?' was a finalist. Provide citations or evidence from authoritative sources for each candidate."
    critic_instruction1 = "Please review the candidate awards and their citations for accuracy and completeness, and provide feedback on any missing or incorrect information."
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, verifying candidate awards, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback on candidate awards, thinking: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining candidate awards, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_sc_instruction2 = "Sub-task 2: Generate multiple independent candidate awards for which the expanded 2008 article was a finalist, ensuring coverage and robustness by comparing and cross-checking each candidate."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.7,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx, ans in enumerate(results2['list_answer']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, generating candidate awards independently, thinking: {results2['list_thinking'][idx]}; answer: {ans}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    debate_instruction3 = "Sub-task 3: Debate the validity, completeness, and evidence supporting each candidate award generated in Sub-task 2 to select the most accurate and complete award name and category."
    final_decision_instruction3 = "Sub-task 3: Make a final decision on the best candidate award based on the debate."
    debate_desc3 = {
        'instruction': debate_instruction3,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"],
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
        subtask_id="subtask_3",
        debate_desc=debate_desc3,
        final_decision_desc=final_decision_desc3,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating candidate awards, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding best candidate award, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    cot_reflect_instruction4 = "Sub-task 4: Critically review and fact-check the aggregated award from Sub-task 3, verifying award category, year, and official title using authoritative references."
    critic_instruction4 = "Please provide detailed fact verification feedback and corrections if necessary."
    cot_reflect_desc4 = {
        'instruction': cot_reflect_instruction4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
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
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, fact-checking aggregated award, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, fact verification feedback, thinking: {results4['list_feedback'][i].content}; correct: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining fact-checked award, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    code_generate_instruction5 = "Sub-task 5: Generate Python code to robustly validate the consolidated award answer, including exact award name, category, and year by querying authoritative databases or APIs."
    code_generate_desc5 = {
        'instruction': code_generate_instruction5,
        'input': [taskInfo, results4['thinking'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4", "answer of subtask 4"],
        'entry_point': 'validate_award'
    }
    results5 = await self.code_generate(
        subtask_id="subtask_5",
        code_generate_desc=code_generate_desc5
    )
    agents.append(f"Code Generate Agent {results5['code_generate_agent'].id}, generating validation code, thinking: {results5['thinking'].content}; code: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; code - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    cot_reflect_instruction6 = "Sub-task 6: Reflect on all previous outputs and validation results to finalize a precise, fully qualified award name (including year and category) for which the expanded 2008 article 'Is Google Making Us Stoopid?' was a finalist. Provide supporting evidence and citations."
    critic_instruction6 = "Please review the final answer for completeness, correctness, and evidence support."
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer'], results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc6 = {
        'instruction': critic_instruction6,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=cot_reflect_desc6,
        critic_desc=critic_desc6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, finalizing award answer, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, reviewing final answer, thinking: {results6['list_feedback'][i].content}; correct: {results6['list_correct'][i].content}")
        if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
            agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final answer, thinking: {results6['list_thinking'][i + 1].content}; answer: {results6['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])
    
    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
