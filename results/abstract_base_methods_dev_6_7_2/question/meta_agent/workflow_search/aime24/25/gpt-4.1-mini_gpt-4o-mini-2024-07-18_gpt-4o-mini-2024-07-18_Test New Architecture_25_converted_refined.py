async def forward_25(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction1 = "Sub-task 1: Precisely define the geometric properties of the convex equilateral hexagon ABCDEF, including the implications of opposite sides being parallel and equal length, and explicitly describe the triangle formed by the extensions of AB, CD, and EF with sides 200, 240, and 300. Use vector or coordinate-based reasoning to relate the triangle's side lengths to the hexagon's side length."
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
    agents.append(f"CoT agent {results1['cot_agent'].id}, defining hexagon properties and triangle formation, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    debate_instruction2 = "Sub-task 2: Debate and verify the assumptions and geometric relationships established in Sub-task 1, challenging the validity of the reasoning and calculations relating the hexagon side length to the triangle sides."
    debate_desc2 = {
        'instruction': debate_instruction2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc2,
        final_decision_desc={
            'instruction': "Sub-task 2: Make final decision on the validity of assumptions and relationships.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating assumptions and relationships, thinking: {results2['list_thinking'][round][idx].content}; answer: {results2['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding on assumptions validity, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction3 = "Sub-task 3: Using multiple reasoning paths, validate the geometric relationships and calculations from Sub-task 2 to ensure consistency and correctness in determining the hexagon side length."
    cot_sc_desc3 = {
        'instruction': cot_sc_instruction3,
        'input': [taskInfo, results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc3,
        n_repeat=self.max_sc
    )
    for idx, agent in enumerate(results3['cot_agent']):
        agents.append(f"CoT-SC agent {agent.id}, validating relationships with multiple reasoning paths, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_reflect_instruction4_1 = "Sub-task 4.1: Reflect on the reasoning and assumptions from previous subtasks to identify and correct any errors before generating candidate hexagon side length solutions."
    critic_instruction4_1 = "Please review the reasoning and assumptions for errors or gaps and provide feedback."
    cot_reflect_desc4_1 = {
        'instruction': cot_reflect_instruction4_1,
        'input': [taskInfo, results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"]
    }
    critic_desc4_1 = {
        'instruction': critic_instruction4_1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4_1 = await self.reflexion(
        subtask_id="subtask_4_1",
        cot_reflect_desc=cot_reflect_desc4_1,
        critic_desc=critic_desc4_1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4_1['cot_agent'].id}, reflecting on reasoning and assumptions, thinking: {results4_1['list_thinking'][0].content}; answer: {results4_1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4_1['list_feedback']))):
        agents.append(f"Critic agent {results4_1['critic_agent'].id}, providing feedback, thinking: {results4_1['list_feedback'][i].content}; answer: {results4_1['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 4.1 output: thinking - {results4_1['thinking'].content}; answer - {results4_1['answer'].content}")
    logs.append(results4_1['subtask_desc'])

    cot_reflect_instruction4_2 = "Sub-task 4.2: Verify the consistency of assumptions and calculations, check geometric constraints, and quantify conditions before finalizing candidate hexagon side length values."
    critic_instruction4_2 = "Please critically evaluate the consistency and correctness of the candidate side length calculations."
    cot_reflect_desc4_2 = {
        'instruction': cot_reflect_instruction4_2,
        'input': [taskInfo, results4_1['thinking'], results4_1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 4_1", "answer of subtask 4_1"]
    }
    critic_desc4_2 = {
        'instruction': critic_instruction4_2,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results4_2 = await self.reflexion(
        subtask_id="subtask_4_2",
        cot_reflect_desc=cot_reflect_desc4_2,
        critic_desc=critic_desc4_2,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results4_2['cot_agent'].id}, verifying consistency and calculations, thinking: {results4_2['list_thinking'][0].content}; answer: {results4_2['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results4_2['list_feedback']))):
        agents.append(f"Critic agent {results4_2['critic_agent'].id}, providing feedback, thinking: {results4_2['list_feedback'][i].content}; answer: {results4_2['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 4.2 output: thinking - {results4_2['thinking'].content}; answer - {results4_2['answer'].content}")
    logs.append(results4_2['subtask_desc'])

    debate_instruction4_3 = "Sub-task 4.3: Debate among agents to critically evaluate and validate the aggregated candidate hexagon side length solutions from previous reflection steps."
    debate_desc4_3 = {
        'instruction': debate_instruction4_3,
        'input': [taskInfo, results4_2['thinking'], results4_2['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 4_2", "answer of subtask 4_2"]
    }
    results4_3 = await self.debate(
        subtask_id="subtask_4_3",
        debate_desc=debate_desc4_3,
        final_decision_desc={
            'instruction': "Sub-task 4.3: Make final decision on validated candidate solutions.",
            'output': ["thinking", "answer"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4_3['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, evaluating candidate solutions, thinking: {results4_3['list_thinking'][round][idx].content}; answer: {results4_3['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding on candidate solutions, thinking: {results4_3['thinking'].content}; answer: {results4_3['answer'].content}")
    sub_tasks.append(f"Sub-task 4.3 output: thinking - {results4_3['thinking'].content}; answer - {results4_3['answer'].content}")
    logs.append(results4_3['subtask_desc'])

    aggregate_instruction5 = "Sub-task 5: Aggregate the validated candidate hexagon side length solutions, evaluate consistency, clarify units, and identify the best consensus solution."
    aggregate_desc5 = {
        'instruction': aggregate_instruction5,
        'input': [taskInfo, results4_3['answer']],
        'temperature': 0.0,
        'context': ["user query", "validated candidate solutions from subtask 4_3"]
    }
    results5 = await self.aggregate(
        subtask_id="subtask_5",
        aggregate_desc=aggregate_desc5
    )
    agents.append(f"Aggregate agent {results5['aggregate_agent'].id}, aggregating candidate solutions, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    cot_reflect_instruction6 = "Sub-task 6: Reflect on the aggregated solution to identify any overlooked details, gaps, or contradictions before finalizing the hexagon side length."
    critic_instruction6 = "Please review the aggregated solution for any inconsistencies or errors and provide feedback."
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
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, reflecting on aggregated solution, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, providing feedback, thinking: {results6['list_feedback'][i].content}; answer: {results6['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    cot_sc_instruction7 = "Sub-task 7: Perform self-consistency checks on the final hexagon side length solution to ensure it is free from ambiguities and errors."
    cot_sc_desc7 = {
        'instruction': cot_sc_instruction7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    results7 = await self.sc_cot(
        subtask_id="subtask_7",
        cot_sc_desc=cot_sc_desc7,
        n_repeat=self.max_sc
    )
    for idx, agent in enumerate(results7['cot_agent']):
        agents.append(f"CoT-SC agent {agent.id}, validating final solution consistency, thinking: {results7['list_thinking'][idx]}; answer: {results7['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    cot_reflect_instruction8 = "Sub-task 8: Critically re-examine and revise the final solution to correct any errors and provide detailed re-derivation or proof."
    critic_instruction8 = "Please review the revised solution for clarity, correctness, and completeness."
    cot_reflect_desc8 = {
        'instruction': cot_reflect_instruction8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    critic_desc8 = {
        'instruction': critic_instruction8,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results8 = await self.reflexion(
        subtask_id="subtask_8",
        cot_reflect_desc=cot_reflect_desc8,
        critic_desc=critic_desc8,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results8['cot_agent'].id}, revising final solution, thinking: {results8['list_thinking'][0].content}; answer: {results8['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results8['list_feedback']))):
        agents.append(f"Critic agent {results8['critic_agent'].id}, providing feedback, thinking: {results8['list_feedback'][i].content}; answer: {results8['list_correct'][i].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    cot_reflect_instruction9 = "Sub-task 9: Refine and clarify the revised solution to enhance coherence and explicitly address any prior ambiguities or errors."
    cot_reflect_desc9 = {
        'instruction': cot_reflect_instruction9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
    }
    results9 = await self.reflexion(
        subtask_id="subtask_9",
        cot_reflect_desc=cot_reflect_desc9,
        critic_desc=None,
        n_repeat=1
    )
    agents.append(f"Reflexion CoT agent {results9['cot_agent'].id}, refining solution clarity, thinking: {results9['thinking'].content}; answer: {results9['answer'].content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    cot_sc_instruction10 = "Sub-task 10: Perform a final self-consistency check on the refined hexagon side length solution to ensure correctness and consistency with the problem requirements."
    cot_sc_desc10 = {
        'instruction': cot_sc_instruction10,
        'input': [taskInfo, results9['thinking'], results9['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 9", "answer of subtask 9"]
    }
    results10 = await self.sc_cot(
        subtask_id="subtask_10",
        cot_sc_desc=cot_sc_desc10,
        n_repeat=self.max_sc
    )
    for idx, agent in enumerate(results10['cot_agent']):
        agents.append(f"CoT-SC agent {agent.id}, final validation of solution, thinking: {results10['list_thinking'][idx]}; answer: {results10['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 10 output: thinking - {results10['thinking'].content}; answer - {results10['answer'].content}")
    logs.append(results10['subtask_desc'])

    final_answer = await self.make_final_answer(results10['thinking'], results10['answer'], sub_tasks, agents)
    return final_answer, logs
