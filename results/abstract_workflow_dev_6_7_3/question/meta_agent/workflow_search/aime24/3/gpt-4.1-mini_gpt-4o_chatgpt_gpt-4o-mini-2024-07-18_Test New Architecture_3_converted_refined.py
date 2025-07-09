async def forward_3(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Subtask 1: Identify and verify critical breakpoints with Reflexion
    reflexion_instruction_1 = "Subtask 1: Identify all critical breakpoints by solving |sin(2πx)|=0.25,0.5 and |cos(3πy)|=0.25,0.5, and verify the correctness of these breakpoints against the problem constraints."
    critic_instruction_1 = "Please review the breakpoint identification and verify the calculations for correctness and completeness."
    reflexion_desc_1 = {
        'instruction': reflexion_instruction_1,
        'input': [taskInfo],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query"]
    }
    critic_desc_1 = {
        'instruction': critic_instruction_1,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results1 = await self.reflexion(
        subtask_id="subtask_1",
        cot_reflect_desc=reflexion_desc_1,
        critic_desc=critic_desc_1,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, identifying and verifying breakpoints, thinking: {results1['list_thinking'][0].content}; answer: {results1['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results1['list_feedback']))):
        agents.append(f"Critic agent {results1['critic_agent'].id}, providing feedback, thinking: {results1['list_feedback'][i].content}; answer: {results1['list_correct'][i].content}")
        if i + 1 < len(results1['list_thinking']) and i + 1 < len(results1['list_answer']):
            agents.append(f"Reflexion CoT agent {results1['cot_agent'].id}, refining breakpoint verification, thinking: {results1['list_thinking'][i + 1].content}; answer: {results1['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    # Subtask 2: Derive explicit piecewise formulas with CoT
    cot_instruction_2 = "Subtask 2: Partition the x- and y-domains at the verified breakpoints and derive explicit piecewise formulas for g(f(sin(2πx))) and g(f(cos(3πy)))."
    cot_agent_desc_2 = {
        'instruction': cot_instruction_2,
        'input': [taskInfo, results1['thinking'], results1['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 1", "answer of subtask 1"]
    }
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_agent_desc=cot_agent_desc_2
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, deriving piecewise formulas, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    # Subtask 3: Apply 4x scaling with Self-Consistency CoT
    cot_sc_instruction_3 = "Subtask 3: Apply the 4× scaling to obtain y=4·g(f(sin(2πx))) and x=4·g(f(cos(3πy))) as explicit piecewise linear functions, verifying multiple reasoning paths for consistency."
    cot_sc_desc_3 = {
        'instruction': cot_sc_instruction_3,
        'input': [taskInfo, results2['thinking'], results2['answer'], results1['thinking'], results1['answer']],
        'temperature': 0.5,
        'context': ["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 1", "answer of subtask 1"]
    }
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_desc=cot_sc_desc_3,
        n_repeat=self.max_sc
    )
    for idx in range(len(results3['list_thinking'])):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, applying 4x scaling, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    # Subtask 4: Generate candidate intersection points with Self-Consistency CoT verifying domain constraints
    candidate_solutions = []
    for i in range(self.max_sc):
        cot_sc_instruction_4 = f"Subtask 4_{i+1}: For each pair of linear branches from y=4·g(f(sin(2πx))) and x=4·g(f(cos(3πy))), solve the linear system to find candidate intersection points and verify each point against domain constraints."
        cot_sc_desc_4 = {
            'instruction': cot_sc_instruction_4,
            'input': [taskInfo, results3['thinking'], results3['answer']],
            'temperature': 0.5,
            'context': ["user query", "thinking of subtask 3", "answer of subtask 3"]
        }
        results4 = await self.sc_cot(
            subtask_id=f"subtask_4_{i+1}",
            cot_sc_desc=cot_sc_desc_4,
            n_repeat=self.max_sc
        )
        for idx in range(len(results4['list_thinking'])):
            agents.append(f"CoT-SC agent {results4['cot_agent'][idx].id}, generating candidate intersections, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
        sub_tasks.append(f"Sub-task 4_{i+1} output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
        logs.append(results4['subtask_desc'])
        candidate_solutions.extend(results4['list_answer'])

    # Subtask 5: Aggregate and debate candidate solutions
    debate_instruction_5 = "Subtask 5: Aggregate all candidate intersection points, debate their validity and consistency with domain constraints, and select the valid intersection points."
    final_decision_instruction_5 = "Subtask 5: Make final decision on the valid intersection points after debate."
    debate_desc_5 = {
        'instruction': debate_instruction_5,
        'context': ["user query", "candidate solutions from subtask 4"],
        'input': [taskInfo] + candidate_solutions,
        'output': ["thinking", "answer"],
        'temperature': 0.5
    }
    final_decision_desc_5 = {
        'instruction': final_decision_instruction_5,
        'output': ["thinking", "answer"],
        'temperature': 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc_5,
        final_decision_desc=final_decision_desc_5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating candidate solutions, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding valid intersections, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    # Subtask 6: Reflexion to verify and refine final intersection points
    reflexion_instruction_6 = "Subtask 6: Reflect on the aggregated intersection points, verify their correctness against problem constraints, and refine the final set of intersection points."
    critic_instruction_6 = "Please review the final intersection points for correctness and consistency with the problem constraints."
    reflexion_desc_6 = {
        'instruction': reflexion_instruction_6,
        'input': [taskInfo, results5['thinking'], results5['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 5", "answer of subtask 5"]
    }
    critic_desc_6 = {
        'instruction': critic_instruction_6,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results6 = await self.reflexion(
        subtask_id="subtask_6",
        cot_reflect_desc=reflexion_desc_6,
        critic_desc=critic_desc_6,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, verifying final intersections, thinking: {results6['list_thinking'][0].content}; answer: {results6['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results6['list_feedback']))):
        agents.append(f"Critic agent {results6['critic_agent'].id}, providing feedback, thinking: {results6['list_feedback'][i].content}; answer: {results6['list_correct'][i].content}")
        if i + 1 < len(results6['list_thinking']) and i + 1 < len(results6['list_answer']):
            agents.append(f"Reflexion CoT agent {results6['cot_agent'].id}, refining final intersections, thinking: {results6['list_thinking'][i + 1].content}; answer: {results6['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    # Subtask 7: Reflexion for rigorous re-derivation and critical review
    reflexion_instruction_7 = "Subtask 7: Critically re-examine all previous steps, verify assumptions, and provide a rigorous justification for the final answer, addressing any external feedback indicating incorrectness."
    critic_instruction_7 = "Please identify any errors, gaps, or oversights in the entire solution process and suggest improvements."
    reflexion_desc_7 = {
        'instruction': reflexion_instruction_7,
        'input': [taskInfo, results6['thinking'], results6['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 6", "answer of subtask 6"]
    }
    critic_desc_7 = {
        'instruction': critic_instruction_7,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results7 = await self.reflexion(
        subtask_id="subtask_7",
        cot_reflect_desc=reflexion_desc_7,
        critic_desc=critic_desc_7,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, re-deriving and reviewing solution, thinking: {results7['list_thinking'][0].content}; answer: {results7['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results7['list_feedback']))):
        agents.append(f"Critic agent {results7['critic_agent'].id}, providing feedback, thinking: {results7['list_feedback'][i].content}; answer: {results7['list_correct'][i].content}")
        if i + 1 < len(results7['list_thinking']) and i + 1 < len(results7['list_answer']):
            agents.append(f"Reflexion CoT agent {results7['cot_agent'].id}, refining final justification, thinking: {results7['list_thinking'][i + 1].content}; answer: {results7['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {results7['thinking'].content}; answer - {results7['answer'].content}")
    logs.append(results7['subtask_desc'])

    # Subtask 8: Reflexion to re-evaluate the final answer
    reflexion_instruction_8 = "Subtask 8: Re-evaluate the final answer based on corrected and verified calculations from previous steps to ensure accuracy."
    critic_instruction_8 = "Please confirm the correctness of the final answer and highlight any remaining issues."
    reflexion_desc_8 = {
        'instruction': reflexion_instruction_8,
        'input': [taskInfo, results7['thinking'], results7['answer']],
        'output': ["thinking", "answer"],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 7", "answer of subtask 7"]
    }
    critic_desc_8 = {
        'instruction': critic_instruction_8,
        'output': ["feedback", "correct"],
        'temperature': 0.0
    }
    results8 = await self.reflexion(
        subtask_id="subtask_8",
        cot_reflect_desc=reflexion_desc_8,
        critic_desc=critic_desc_8,
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results8['cot_agent'].id}, re-evaluating final answer, thinking: {results8['list_thinking'][0].content}; answer: {results8['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results8['list_feedback']))):
        agents.append(f"Critic agent {results8['critic_agent'].id}, providing feedback, thinking: {results8['list_feedback'][i].content}; answer: {results8['list_correct'][i].content}")
        if i + 1 < len(results8['list_thinking']) and i + 1 < len(results8['list_answer']):
            agents.append(f"Reflexion CoT agent {results8['cot_agent'].id}, refining final answer, thinking: {results8['list_thinking'][i + 1].content}; answer: {results8['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {results8['thinking'].content}; answer - {results8['answer'].content}")
    logs.append(results8['subtask_desc'])

    # Subtask 9: Reflexion for final verification and answer generation
    reflexion_instruction_9 = "Subtask 9: Perform final verification of the answer ensuring it is based on verified and correct calculations, then generate the final integer count of intersections."
    cot_agent_desc_9 = {
        'instruction': reflexion_instruction_9,
        'input': [taskInfo, results8['thinking'], results8['answer']],
        'temperature': 0.0,
        'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
    }
    results9 = await self.reflexion(
        subtask_id="subtask_9",
        cot_reflect_desc={
            'instruction': reflexion_instruction_9,
            'input': [taskInfo, results8['thinking'], results8['answer']],
            'output': ["thinking", "answer"],
            'temperature': 0.0,
            'context': ["user query", "thinking of subtask 8", "answer of subtask 8"]
        },
        critic_desc={
            'instruction': "Please verify the final answer for correctness.",
            'output': ["feedback", "correct"],
            'temperature': 0.0
        },
        n_repeat=self.max_round
    )
    agents.append(f"Reflexion CoT agent {results9['cot_agent'].id}, final verification and answer generation, thinking: {results9['list_thinking'][0].content}; answer: {results9['list_answer'][0].content}")
    for i in range(min(self.max_round, len(results9['list_feedback']))):
        agents.append(f"Critic agent {results9['critic_agent'].id}, providing feedback, thinking: {results9['list_feedback'][i].content}; answer: {results9['list_correct'][i].content}")
        if i + 1 < len(results9['list_thinking']) and i + 1 < len(results9['list_answer']):
            agents.append(f"Reflexion CoT agent {results9['cot_agent'].id}, refining final answer, thinking: {results9['list_thinking'][i + 1].content}; answer: {results9['list_answer'][i + 1].content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {results9['thinking'].content}; answer - {results9['answer'].content}")
    logs.append(results9['subtask_desc'])

    final_answer = await self.make_final_answer(results9['thinking'], results9['answer'], sub_tasks, agents)
    return final_answer, logs
