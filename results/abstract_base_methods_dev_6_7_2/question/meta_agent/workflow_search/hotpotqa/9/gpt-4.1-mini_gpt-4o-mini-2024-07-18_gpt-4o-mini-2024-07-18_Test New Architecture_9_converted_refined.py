async def forward_9(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_reflect_instruction1 = "Sub-task 1: Identify the Shakespeare tragedy that includes the fictional character Benvolio and determine if Benvolio slays a character in it. Then validate the premise that Benvolio slays anyone, explicitly addressing any contradictions with known facts."
    critic_instruction1 = "Please review the identification and premise validation about Benvolio slaying a character and provide any contradictions or errors."
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, identifying tragedy with Benvolio and validating premise, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, providing feedback, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining answer, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_instruction2 = "Sub-task 2: List possible protagonists of the Shakespeare tragedy identified in Sub-task 1."
    cot_agent_desc2 = {
        'instruction': cot_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, listing protagonists, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    protagonists = results2['answer'].content if hasattr(results2['answer'], 'content') else results2['answer']
    if isinstance(protagonists, str):
        protagonists_list = [p.strip() for p in protagonists.split(',') if p.strip()]
    else:
        protagonists_list = protagonists
    protagonist_love_marriage_results = []
    for idx, protagonist in enumerate(protagonists_list, start=3):
        cot_instruction3 = f"Sub-task {idx}: Check if protagonist {protagonist} secretly loves and marries a member of a rival house in the identified tragedy."
        cot_agent_desc3 = {
            'instruction': cot_instruction3,
            'input': [taskInfo, protagonist, results1['answer'].content],
            'temperature': 0.0,
            'context': ["user query", "protagonist info", "tragedy info"]
        }
        results3 = await self.cot(
            subtask_id=f"subtask_{idx}",
            cot_agent_desc=cot_agent_desc3
        )
        agents.append(f"CoT agent {results3['cot_agent'].id}, checking love and marriage for {protagonist}, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
        sub_tasks.append(f"Sub-task {idx} output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
        logs.append(results3['subtask_desc'])
        protagonist_love_marriage_results.append((protagonist, results3['answer'].content))
    debate_instruction4 = "Sub-task 4: Debate the premise that Benvolio slays a character in the identified Shakespeare tragedy, and discuss the candidate protagonists and their relationships to rival houses. Explicitly challenge or confirm the question's premise and reconcile contradictions."
    final_decision_instruction4 = "Sub-task 4: Make a final decision on the validity of the premise and the correct protagonist based on the debate."
    debate_desc4 = {
        'instruction': debate_instruction4,
        'context': ["user query", results1['thinking'].content, results1['answer'].content] + [res[1] for res in protagonist_love_marriage_results],
        'input': [taskInfo, results1['thinking'], results1['answer']] + [res[1] for res in protagonist_love_marriage_results],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc4 = {
        'instruction': final_decision_instruction4,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc4,
        final_decision_desc=final_decision_desc4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating premise and protagonists, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, concluding debate, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    cot_reflect_instruction5 = "Sub-task 5: Reflect on the entire workflow outputs to verify consistency between the question, the premise about Benvolio, and the consolidated answer. If contradictions exist, revise the final answer to explicitly state the correct interpretation."
    critic_instruction5 = "Please review the final consolidated answer for consistency and correctness, especially regarding the premise about Benvolio slaying a character."
    cot_reflect_desc5 = {
        'instruction': cot_reflect_instruction5,
        'input': [taskInfo, results1['thinking'], results1['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 4", "answer of subtask 4"]
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
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, validating final answer, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, providing feedback, thinking: {results5['list_feedback'][i].content}; answer: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
