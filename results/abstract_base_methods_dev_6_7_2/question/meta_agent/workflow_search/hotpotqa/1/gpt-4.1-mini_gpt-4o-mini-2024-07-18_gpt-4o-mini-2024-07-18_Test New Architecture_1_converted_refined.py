async def forward_1(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Enumerate and verify Jonny Craig's band memberships with Reflexion
    cot_reflect_instruction1 = "Sub-task 1: Enumerate all bands Jonny Craig has been a member of, excluding solo projects and collaborations that do not qualify as formal bands. Verify each band's existence and membership with context from taskInfo."
    critic_instruction1 = "Please review the enumeration of Jonny Craig's bands and provide feedback on any assumptions or errors."
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, enumerating Jonny Craig's bands, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correction: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining answer, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    # Stage 2: Enumerate and verify Pete Doherty's band memberships with Self-Consistency CoT
    cot_sc_instruction2 = "Sub-task 2: Enumerate all bands Pete Doherty has been a member of, excluding solo projects and ambiguous collaborations. Explore multiple reasoning paths to verify band memberships with context from taskInfo."
    cot_sc_desc2 = {
        'instruction': cot_sc_instruction2,
        'input': [taskInfo],
        'temperature': 0.5,
        'context': ["user query"]
    }
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results2['list_thinking']):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, enumerating Pete Doherty's bands, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    # Stage 3: Stepwise numeric count of Jonny Craig's bands with CoT
    cot_instruction3 = "Sub-task 3: Provide a concise numeric count of Jonny Craig's band memberships based on the enumerated list from Sub-task 1, excluding solo projects and collaborations that do not qualify as bands."
    cot_agent_desc3 = {
        'instruction': cot_instruction3,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_agent_desc=cot_agent_desc3
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, counting Jonny Craig's bands, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    # Stage 4: Stepwise numeric count of Pete Doherty's bands with CoT
    cot_instruction4 = "Sub-task 4: Provide a concise numeric count of Pete Doherty's band memberships based on the enumerated lists from Sub-task 2, excluding solo projects and ambiguous collaborations."
    cot_agent_desc4 = {
        'instruction': cot_instruction4,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results4 = await self.cot(
        subtask_id="subtask_4",
        cot_agent_desc=cot_agent_desc4
    )
    agents.append(f"CoT agent {results4['cot_agent'].id}, counting Pete Doherty's bands, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    # Stage 5: Consolidate and cross-check band counts with Self-Consistency CoT
    cot_sc_instruction5 = "Sub-task 5: Aggregate and cross-check the numeric band counts of Jonny Craig and Pete Doherty from Sub-tasks 3 and 4. Identify inconsistencies and determine who has been a member of more bands with justification."
    cot_sc_desc5 = {
        'instruction': cot_sc_instruction5,
        'input': [taskInfo, results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 3", "answer of subtask 3", "thinking of subtask 4", "answer of subtask 4"]
    }
    results5 = await self.sc_cot(
        subtask_id="subtask_5",
        cot_sc_desc=cot_sc_desc5,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results5['list_thinking']):
        agents.append(f"CoT-SC agent {results5['cot_agent'][idx].id}, consolidating band counts, thinking: {results5['list_thinking'][idx]}; answer: {results5['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    # Stage 6: Final validation and fact-checking with Reflexion
    cot_reflect_instruction6 = "Sub-task 6: Verify the consolidated answer from Sub-task 5 against external knowledge or reliable sources. Revise the conclusion if discrepancies or insufficient evidence are found."
    critic_instruction6 = "Please review the consolidated band membership comparison and provide feedback on its accuracy and any necessary corrections."
    cot_reflect_desc6 = {
        'instruction': cot_reflect_instruction6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
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
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, validating consolidated answer, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, feedback: {results6['list_feedback'][i].content}; correction: {results6['list_correct'][i].content}")
        if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
            agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final answer, thinking: {results6['list_thinking'][i + 1].content}; answer: {results6['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
