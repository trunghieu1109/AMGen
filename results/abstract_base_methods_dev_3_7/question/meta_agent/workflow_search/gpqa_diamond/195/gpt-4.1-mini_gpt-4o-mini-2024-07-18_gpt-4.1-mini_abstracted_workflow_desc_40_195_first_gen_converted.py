async def forward_195(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Extract and define the physical parameters and constraints of the 1D relativistic harmonic oscillator, including mass m, amplitude A, spring constant k, speed of light c, and clarify the force law (Hooke's law: F = -kx)."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC_CoT agent {results1['cot_agent'][idx].id}, extracting parameters, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = "Subtask 2: Analyze the classical harmonic oscillator energy relations (potential and kinetic energy) and express the classical maximum speed in terms of k, A, and m for baseline comparison, based on outputs from Subtask 1."
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_instruction_2,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC_CoT agent {results2['cot_agent'][idx].id}, analyzing classical energy, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction_3 = "Subtask 3: Analyze the relativistic energy relations for the oscillator, incorporating relativistic kinetic energy and total energy, and express total energy in terms of rest mass energy, potential energy, and relativistic kinetic energy, based on outputs from Subtask 1."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_instruction_3,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC_CoT agent {results3['cot_agent'][idx].id}, analyzing relativistic energy, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    cot_instruction_4 = "Subtask 4: Derive the expression for the maximum speed v_max of the mass in the relativistic harmonic oscillator by applying energy conservation and relativistic velocity-energy relations, using parameters m, k, A, and c, based on outputs from Subtasks 2 and 3."
    results4 = await self.sc_cot(
        subtask_id="subtask_4",
        cot_sc_instruction=cot_instruction_4,
        input_list=[taskInfo, results2['thinking'], results2['answer'], results3['thinking'], results3['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"SC_CoT agent {results4['cot_agent'][idx].id}, deriving v_max relativistic expression, thinking: {results4['list_thinking'][idx]}; answer: {results4['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction_5 = "Subtask 5: Evaluate the four given multiple-choice expressions for v_max against the derived relativistic expression to identify which choice correctly represents the maximum speed, based on output from Subtask 4."
    final_decision_instruction_5 = "Subtask 5: Make final decision on the correct multiple-choice formula for v_max."
    debate_desc_5 = {
        "instruction": debate_instruction_5,
        "context": ["user query", results4['thinking'], results4['answer']],
        "input": [taskInfo, results4['thinking'], results4['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_5 = {
        "instruction": final_decision_instruction_5,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results5 = await self.debate(
        subtask_id="subtask_5",
        debate_desc=debate_desc_5,
        final_decision_desc=final_decision_desc_5,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results5['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, evaluating choices, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, deciding correct choice, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    debate_instruction_6 = "Subtask 6: Select and return the alphabet choice (A, B, C, or D) corresponding to the correct formula for v_max based on the evaluation in Subtask 5."
    final_decision_instruction_6 = "Subtask 6: Finalize and return the alphabet choice as the answer."
    debate_desc_6 = {
        "instruction": debate_instruction_6,
        "context": ["user query", results5['thinking'], results5['answer']],
        "input": [taskInfo, results5['thinking'], results5['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_6 = {
        "instruction": final_decision_instruction_6,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results6 = await self.debate(
        subtask_id="subtask_6",
        debate_desc=debate_desc_6,
        final_decision_desc=final_decision_desc_6,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results6['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, selecting final choice, thinking: {results6['list_thinking'][round][idx].content}; answer: {results6['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, finalizing answer, thinking: {results6['thinking'].content}; answer: {results6['answer'].content}")
    sub_tasks.append(f"Subtask 6 output: thinking - {results6['thinking'].content}; answer - {results6['answer'].content}")
    logs.append(results6['subtask_desc'])

    final_answer = await self.make_final_answer(results6['thinking'], results6['answer'], sub_tasks, agents)
    return final_answer, logs
