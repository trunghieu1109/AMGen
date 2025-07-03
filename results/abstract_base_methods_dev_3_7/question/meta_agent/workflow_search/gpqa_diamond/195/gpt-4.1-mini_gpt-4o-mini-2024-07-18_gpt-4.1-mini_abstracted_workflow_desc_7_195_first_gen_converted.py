async def forward_195(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Analyze the classical (non-relativistic) harmonic oscillator to find the expression for maximum speed v_max in terms of k, m, and A, with context from the user query."
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user input"
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, analyzing classical harmonic oscillator, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    debate_instruction_1 = "Subtask 1 Debate: Debate the classical maximum speed expression derived in Subtask 1 to verify correctness and clarify assumptions."
    debate_desc_1 = {
        "instruction": debate_instruction_1,
        "context": ["user query", results1['thinking'].content, results1['answer'].content],
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    debate_results_1 = await self.debate(
        subtask_id="subtask_1_debate",
        debate_desc=debate_desc_1,
        final_decision_desc={
            "instruction": "Subtask 1 Debate: Make final decision on the classical maximum speed expression.",
            "output": ["thinking", "answer"],
            "temperature": 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_results_1['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating classical max speed, thinking: {debate_results_1['list_thinking'][round][idx].content}; answer: {debate_results_1['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, classical max speed debate, thinking: {debate_results_1['thinking'].content}; answer: {debate_results_1['answer'].content}")
    sub_tasks.append(f"Subtask 1 Debate output: thinking - {debate_results_1['thinking'].content}; answer - {debate_results_1['answer'].content}")
    logs.append(debate_results_1['subtask_desc'])

    cot_instruction_2 = "Subtask 2: Review relativistic dynamics principles relevant to a 1D oscillator, including relativistic momentum and energy relations, to understand how velocity relates to energy and force, with context from the user query."
    results2 = await self.cot(
        subtask_id="subtask_2",
        cot_instruction=cot_instruction_2,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user input"
    )
    agents.append(f"CoT agent {results2['cot_agent'].id}, reviewing relativistic dynamics, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")

    cot_sc_instruction_2 = "Subtask 2 SC-CoT: Based on the output from Subtask 2 CoT, consider multiple cases and self-consistent reasoning about relativistic momentum and energy relations for the oscillator."
    results2_sc = await self.sc_cot(
        subtask_id="subtask_2_sc",
        cot_sc_instruction=cot_sc_instruction_2,
        input_list=[taskInfo, results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results2['thinking'].content, results2['answer'].content],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2_sc['cot_agent'][idx].id}, considering relativistic dynamics cases, thinking: {results2_sc['list_thinking'][idx]}; answer: {results2_sc['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2_sc['list_thinking'][0]}; answer - {results2_sc['list_answer'][0]}")
    logs.append(results2_sc['subtask_desc'])

    cot_instruction_3 = "Subtask 3: Derive the relativistic expression for total energy of the oscillator at maximum amplitude and relate it to kinetic and potential energy, incorporating Hooke's law and relativistic mass-energy equivalence, based on outputs from Subtask 1 and Subtask 2."
    cot_sc_instruction_3 = "Subtask 3 SC-CoT: Consider multiple self-consistent derivations of the relativistic total energy expression for the oscillator."
    results3_cot = await self.cot(
        subtask_id="subtask_3_cot",
        cot_instruction=cot_instruction_3,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results2_sc['list_thinking'][0], results2_sc['list_answer'][0]],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", results1['thinking'].content, results1['answer'].content, results2_sc['list_thinking'][0], results2_sc['list_answer'][0]]
    )
    results3_sc = await self.sc_cot(
        subtask_id="subtask_3_sc",
        cot_sc_instruction=cot_sc_instruction_3,
        input_list=[taskInfo, results1['thinking'], results1['answer'], results2_sc['list_thinking'][0], results2_sc['list_answer'][0], results3_cot['thinking'], results3_cot['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results1['thinking'].content, results1['answer'].content, results2_sc['list_thinking'][0], results2_sc['list_answer'][0], results3_cot['thinking'].content, results3_cot['answer'].content],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results3_sc['cot_agent'][idx].id}, deriving relativistic total energy, thinking: {results3_sc['list_thinking'][idx]}; answer: {results3_sc['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3_sc['list_thinking'][0]}; answer - {results3_sc['list_answer'][0]}")
    logs.append(results3_sc['subtask_desc'])

    cot_instruction_4 = "Subtask 4: From the relativistic energy expression derived in Subtask 3, derive the formula for maximum speed v_max of the mass, expressing it in terms of k, m, A, and c."
    debate_instruction_4 = "Subtask 4 Debate: Debate the derived relativistic maximum speed formula to verify correctness and consistency."
    results4_cot = await self.cot(
        subtask_id="subtask_4_cot",
        cot_instruction=cot_instruction_4,
        input_list=[taskInfo, results3_sc['list_thinking'][0], results3_sc['list_answer'][0]],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", results3_sc['list_thinking'][0], results3_sc['list_answer'][0]]
    )
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", results4_cot['thinking'].content, results4_cot['answer'].content],
        "input": [taskInfo, results4_cot['thinking'], results4_cot['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    debate_results_4 = await self.debate(
        subtask_id="subtask_4_debate",
        debate_desc=debate_desc_4,
        final_decision_desc={
            "instruction": "Subtask 4 Debate: Make final decision on the relativistic maximum speed formula.",
            "output": ["thinking", "answer"],
            "temperature": 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(debate_results_4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, debating relativistic max speed, thinking: {debate_results_4['list_thinking'][round][idx].content}; answer: {debate_results_4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, relativistic max speed debate, thinking: {debate_results_4['thinking'].content}; answer: {debate_results_4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {debate_results_4['thinking'].content}; answer - {debate_results_4['answer'].content}")
    logs.append(debate_results_4['subtask_desc'])

    debate_instruction_5 = "Subtask 5: Compare the derived relativistic maximum speed formula with the given multiple-choice options to identify the correct choice."
    final_decision_instruction_5 = "Subtask 5: Make final decision on the correct multiple-choice option for the maximum speed v_max."
    debate_desc_5 = {
        "instruction": debate_instruction_5,
        "context": ["user query", debate_results_4['thinking'].content, debate_results_4['answer'].content],
        "input": [taskInfo, debate_results_4['thinking'], debate_results_4['answer']],
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
            agents.append(f"Debate agent {agent.id}, round {round}, comparing derived formula with choices, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
