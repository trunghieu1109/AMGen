async def forward_0(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_reflect_instruction1 = "Sub-task 1: Carefully analyze the problem statement to correctly interpret the walking times, speeds, and coffee break time t, and formulate the correct algebraic equations representing the two scenarios. Validate the interpretation against the problem context and check for consistency."
    critic_instruction1 = "Please review the interpretation and equations for correctness and identify any subtle errors or misinterpretations."
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
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, analyzing problem interpretation, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, feedback: {results1['list_feedback'][i].content}; correct: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining interpretation, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    debate_instruction2 = "Sub-task 2: Debate the validity of the algebraic equations derived from the problem interpretation in Sub-task 1. Challenge the assumptions and propose alternative formulations if necessary to ensure the equations accurately model the walking times, speeds, and coffee break time t."
    final_decision_instruction2 = "Sub-task 2: Make a final decision on the correct set of algebraic equations modeling the problem."
    debate_desc2 = {
        'instruction': debate_instruction2,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"],
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc2 = {
        'instruction': final_decision_instruction2,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        final_decision_desc=final_decision_desc2,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating equations, thinking: {results2['list_thinking'][round][idx].content}; answer: {results2['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding correct equations, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_reflect_instruction3 = "Sub-task 3: Solve the algebraic equations decided in Sub-task 2 to find the walking speed s (km/h) and coffee break time t (minutes). Reflect on the arithmetic and unit conversions to ensure the total time calculations are accurate and reasonable."
    critic_instruction3 = "Please review the algebraic solution and calculations for correctness, especially the total time computation and unit consistency."
    cot_reflect_desc3 = {
        'instruction': cot_reflect_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
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
    agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, solving equations, thinking: {results3['list_thinking'][0].content}; answer: {results3['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results3['list_feedback']))):
        agents.append(f"Critic agent {results3['critic_agent'].id}, feedback: {results3['list_feedback'][i].content}; correct: {results3['list_correct'][i].content}")
        if i + 1 < len(results3['list_thinking']) and i + 1 < len(results3['list_answer']):
            agents.append(f"Reflexion CoT agent {results3['cot_agent'].id}, refining solution, thinking: {results3['list_thinking'][i + 1].content}; answer: {results3['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_reflect_instruction4 = "Sub-task 4: Validate the solution for s and t by generating and running code that checks the equations and total time calculations, ensuring the solution satisfies the problem constraints within reasonable tolerance. Return numeric total time in minutes for the original scenarios."
    critic_instruction4 = "Please review the validation code and results, ensuring correctness and clarity in output."
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
    agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, validating solution, thinking: {results4['list_thinking'][0].content}; answer: {results4['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4['list_feedback']))):
        agents.append(f"Critic agent {results4['critic_agent'].id}, feedback: {results4['list_feedback'][i].content}; correct: {results4['list_correct'][i].content}")
        if i + 1 < len(results4['list_thinking']) and i + 1 < len(results4['list_answer']):
            agents.append(f"Reflexion CoT agent {results4['cot_agent'].id}, refining validation, thinking: {results4['list_thinking'][i + 1].content}; answer: {results4['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_instruction5 = "Sub-task 5: Compare the validated solution with previous results, explain any corrections or improvements made, and finalize the values of s and t for further calculations."
    cot_agent_desc5 = {
        'instruction': cot_instruction5,
        'input': [taskInfo, results3['answer'], results4['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 3", "answer of subtask 4"]
    }
    results5 = await self.cot(
        subtask_id="subtask_5",
        cot_agent_desc=cot_agent_desc5
    )
    agents.append(f"CoT agent {results5['cot_agent'].id}, comparing and revising solutions, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_instruction6 = "Sub-task 6: Convert all times and speeds into consistent units (hours and kilometers), explicitly checking unit consistency and correctness of conversions for further calculations."
    cot_agent_desc6 = {
        'instruction': cot_instruction6,
        'input': [taskInfo, results5['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 5"]
    }
    results6 = await self.cot(
        subtask_id="subtask_6",
        cot_agent_desc=cot_agent_desc6
    )
    agents.append(f"CoT agent {results6['cot_agent'].id}, converting units, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_instruction7 = "Sub-task 7: Compute the quantitative relationship between the increase in walking speed by 0.5 km/h and the resulting total walking time including coffee break, using explicit algebraic manipulation and interpretation of results."
    cot_agent_desc7 = {
        'instruction': cot_instruction7,
        'input': [taskInfo, results6['answer']],
        'temperature': 0.0,
        'context': ["user query", "answer of subtask 6"]
    }
    results7 = await self.cot(
        subtask_id="subtask_7",
        cot_agent_desc=cot_agent_desc7
    )
    agents.append(f"CoT agent {results7['cot_agent'].id}, computing speed-time relationship, thinking: {results7['thinking'].content}; answer: {results7['answer'].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_sc_instruction8 = "Sub-task 8: Using self-consistency, derive the total number of minutes the walk takes Aya at speed s + 0.5 km/h including the coffee break time t, verifying correctness by cross-checking with initial equations and previous results."
    cot_sc_desc8 = {
        'instruction': cot_sc_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer'], results6['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7", "answer of subtask 6"]
    }
    results8 = await self.sc_cot(
        subtask_id="subtask_8",
        cot_sc_desc=cot_sc_desc8,
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results8['list_thinking']):
        agents.append(f"CoT-SC agent {results8['cot_agent'][idx].id}, deriving total minutes at s+0.5 km/h, thinking: {results8['list_thinking'][idx]}; answer: {results8['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    final_answer = await self.make_final_answer(results8['thinking'], results8['answer'], sub_tasks, agents)
    return final_answer, logs
