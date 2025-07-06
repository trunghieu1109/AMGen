async def forward_47(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    
    cot_reflect_instruction1 = (
        "Sub-task 1: Reflect on the question's ambiguity regarding 'second touchdown'. "
        "Enumerate all touchdowns mentioned in the passage in order, specifying the team and yardage, "
        "and confirm whether 'second touchdown' refers to the second overall touchdown or second by a specific team before extracting yardage."
    )
    critic_instruction1 = (
        "Please review the enumeration and clarification of 'second touchdown' meaning, "
        "and provide feedback on clarity, completeness, and correctness of the enumeration and interpretation."
    )
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, reflecting on question ambiguity and enumerating touchdowns, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, providing feedback, thinking: {results1['list_feedback'][k].content}; answer: {results1['list_correct'][k].content}")
        if k + 1 < len(results1['list_thinking']) and k + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining enumeration and clarification, thinking: {results1['list_thinking'][k + 1].content}; answer: {results1['list_answer'][k + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    
    cot_instruction2 = (
        "Sub-task 2: Based on the clarified definition of 'second touchdown' from Sub-task 1, "
        "list all touchdowns in the passage chronologically with their yardages and teams, "
        "then identify the yardage of the second touchdown overall."
    )
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
    agents.append(f"CoT agent {results2['cot_agent'].id}, listing touchdowns chronologically and identifying second touchdown yardage, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    
    debate_instruction3 = (
        "Sub-task 3: Review and debate the answer from Sub-task 2, "
        "challenging the interpretation of 'second touchdown' and the correctness of the yardage identified. "
        "Each debater must justify acceptance or rejection with reference to the passage and question intent."
    )
    final_decision_instruction3 = "Sub-task 3: Make a final decision on the correctness and reliability of the identified second touchdown yardage."
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
            agents.append(f"Debate agent {agent.id}, round {round}, debating interpretation and correctness, thinking: {results3['list_thinking'][round][idx].content}; answer: {results3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, concluding on second touchdown yardage, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    
    cot_reflect_instruction4 = (
        "Sub-task 4: Re-validate and refine the answer from Sub-task 3 in light of debate feedback, "
        "re-examining the passage and question to ensure the answer correctly addresses the intended meaning of 'second touchdown'."
    )
    critic_instruction4 = (
        "Please review the refined answer for any remaining ambiguity or errors, "
        "and provide feedback for further improvement if needed."
    )
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
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, re-validating answer based on debate feedback, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for k in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, providing feedback, thinking: {results4['list_feedback'][k].content}; answer: {results4['list_correct'][k].content}")
        if k + 1 < len(results4['list_thinking']) and k + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining final answer, thinking: {results4['list_thinking'][k + 1].content}; answer: {results4['list_answer'][k + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    
    cot_sc_instruction5 = (
        "Sub-task 5: Aggregate all refined answers from previous subtasks, "
        "considering correctness confidence and consistency with the clarified question interpretation, "
        "to select the most reliable yardage for the second touchdown."
    )
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking and answers of subtasks 2, 3, and 4"]
    }
    results5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    for idx, agent in enumerate(results5['cot_agent']):
        agents.append(f"CoT-SC agent {agent.id}, aggregating refined answers with confidence, thinking: {results5['list_thinking'][idx]}; answer: {results5['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])
    
    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
