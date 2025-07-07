async def forward_14(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_0 = "Sub-task 0: Identify and extract the defining features of the problem including the hyperbola equation, rhombus properties, and the condition that the diagonals intersect at the origin. Explicitly list these features and their implications for parameterization and subsequent subtasks."
    cot_agent_desc_0 = {
        'instruction': cot_instruction_0,
        'input': [taskInfo],
        'temperature': 0.0,
        'context': ["user query"]
    }
    results0 = await self.cot(
        subtask_id="subtask_0",
        cot_agent_desc=cot_agent_desc_0
    )
    agents.append(f"CoT agent {results0['cot_agent'].id}, extracting defining features, thinking: {results0['thinking'].content}; answer: {results0['answer'].content}")
    sub_tasks.append(f"Sub-task 0 output: thinking - {results0['thinking'].content}; answer - {results0['answer'].content}")
    logs.append(results0['subtask_desc'])

    cot_reflect_instruction_1 = "Sub-task 1: Parameterize points C and D on the hyperbola such that they reflect the rhombus properties correctly, specifically that C and D are the negatives of A and B respectively, and verify this parameterization aligns with the rhombus conditions including diagonals intersecting at the origin."
    critic_instruction_1 = "Please review the parameterization of points C and D and provide feedback on correctness and alignment with rhombus properties."
    cot_reflect_desc_1 = {
        'instruction': cot_reflect_instruction_1,
        'input': [taskInfo, results0['thinking'], results0['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 0", "answer of subtask 0"]
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=cot_reflect_desc_1,
        critic_desc=critic_desc_1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, verifying parameterization of points C and D, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, providing feedback, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining parameterization, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    debate_instruction_2 = "Sub-task 2: Fully derive and reconcile the rhombus condition equations including side equality and diagonal perpendicularity, explicitly deriving the orthogonality condition x1*x2 + y1*y2 = 0 and ensuring all parameterizations are consistent."
    final_decision_instruction_2 = "Sub-task 2: Make final decision on the derived rhombus condition equations and constraints."
    debate_desc_2 = {
        'instruction': debate_instruction_2,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1", "answer of subtask 1"]
    }
    final_decision_desc_2 = {
        'instruction': final_decision_instruction_2,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc=debate_desc_2,
        final_decision_desc=final_decision_desc_2,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, deriving rhombus conditions, thinking: {results2['list_thinking'][round][idx].content}; answer: {results2['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding on rhombus conditions, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_sc_instruction_3 = "Sub-task 3: Generate multiple candidate parameter pairs (t1, t2) for the rhombus vertices, compute BD^2 values, verify rhombus side equality and diagonal perpendicularity explicitly for each candidate, and cross-validate results to identify consistent solutions."
    cot_sc_desc_3 = {
        'instruction': cot_sc_instruction_3,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer'], results2['thinking'], results2['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 0", "answer of subtask 0", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc_3,
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, generating and verifying candidate parameters, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = "Sub-task 4: Aggregate all valid BD^2 candidate values, evaluate their consistency against rhombus conditions, propose and defend different aggregation strategies, and select the highest lower-bound (supremum) supported by all valid candidates."
    final_decision_instruction_4 = "Sub-task 4: Make final decision on the aggregated supremum BD^2 value and validate correctness."
    debate_desc_4 = {
        'instruction': debate_instruction_4,
        'input': [taskInfo, results3['thinking'], results3['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.5,
        'context': ["user query", "candidate BD^2 values from subtask 3"]
    }
    final_decision_desc_4 = {
        'instruction': final_decision_instruction_4,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc=final_decision_desc_4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, aggregating BD^2 candidates, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding on supremum BD^2, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    cot_reflect_instruction_5 = "Sub-task 5: Reflect on the entire search and aggregation process to identify any overlooked parameterizations, gaps, or ambiguities, and refine the final answer accordingly."
    critic_instruction_5 = "Please review the search process and final answer for completeness and identify any missing configurations or parameterizations."
    cot_reflect_desc_5 = {
        'instruction': cot_reflect_instruction_5,
        'input': [taskInfo, results0['thinking'], results0['answer'], results1['thinking'], results1['answer'], results2['thinking'], results2['answer'], results3['thinking'], results3['answer'], results4['thinking'], results4['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking and answers of all previous subtasks"]
    }
    critic_desc_5 = {
        'instruction': critic_instruction_5,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results5 = await self.reflexion(
        subtask_id="subtask_5",
        cot_reflect_desc=cot_reflect_desc_5,
        critic_desc=critic_desc_5,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, reflecting on search process and final answer, thinking: {results5['list_thinking'][0].content}; answer: {results5['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results5['list_feedback']))):
        agents.append(f"Critic agent {results5['critic_agent'].id}, providing feedback, thinking: {results5['list_feedback'][i].content}; answer: {results5['list_correct'][i].content}")
        if i + 1 < len(results5['list_thinking']) and i + 1 < len(results5['list_answer']):
            agents.append(f"Reflexion CoT agent {results5['cot_agent'].id}, refining final answer, thinking: {results5['list_thinking'][i + 1].content}; answer: {results5['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
