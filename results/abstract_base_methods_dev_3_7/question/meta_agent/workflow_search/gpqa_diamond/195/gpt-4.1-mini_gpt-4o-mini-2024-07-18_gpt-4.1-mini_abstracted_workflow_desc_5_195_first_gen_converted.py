async def forward_195(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []
    cot_instruction_1 = "Subtask 1: Extract and define the physical parameters and setup of the 1D relativistic harmonic oscillator, including mass m, amplitude A, spring constant k, and the force law F = -kx. Identify the relevant physical laws and constraints, including relativistic effects and speed of light c."
    results1 = await self.sc_cot(
        subtask_id="subtask_1",
        cot_sc_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query"],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results1['list_thinking']):
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, extracting physical parameters and setup, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])
    cot_instruction_2 = "Subtask 2: Analyze the energy relations for the relativistic harmonic oscillator, including kinetic energy, potential energy, and total energy, incorporating relativistic corrections (relativistic kinetic energy and momentum), based on the parameters extracted in Subtask 1."
    results2 = await self.debate(
        subtask_id="subtask_2",
        debate_desc={
            "instruction": cot_instruction_2,
            "context": ["user query", results1['thinking'].content, results1['answer'].content],
            "input": [taskInfo, results1['thinking'], results1['answer']],
            "output": ["thinking", "answer"],
            "temperature": 0.5
        },
        final_decision_desc={
            "instruction": "Subtask 2: Make final decision on the energy relations analysis.",
            "output": ["thinking", "answer"],
            "temperature": 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results2['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, analyzing energy relations, thinking: {results2['list_thinking'][round][idx].content}; answer: {results2['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, energy relations analysis, thinking: {results2['thinking'].content}; answer: {results2['answer'].content}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])
    cot_instruction_3 = "Subtask 3: Derive the expression for the maximum speed v_max of the mass by applying conservation of energy and relativistic velocity-momentum relations, using the parameters and energy expressions from previous subtasks."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_instruction_3,
        input_list=[taskInfo, results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", results2['thinking'].content, results2['answer'].content],
        n_repeat=self.max_sc
    )
    for idx, key in enumerate(results3['list_thinking']):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, deriving v_max expression, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])
    debate_instruction_4 = "Subtask 4: Compare the derived expression for v_max with the given multiple-choice options, analyze each choice for physical correctness and consistency with the derived formula, and select the correct choice (A, B, C, or D)."
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc={
            "instruction": debate_instruction_4,
            "context": ["user query", results3['thinking'].content, results3['answer'].content],
            "input": [taskInfo, results3['thinking'], results3['answer']],
            "output": ["thinking", "answer"],
            "temperature": 0.5
        },
        final_decision_desc={
            "instruction": "Subtask 4: Make final decision on the correct multiple-choice option for v_max.",
            "output": ["thinking", "answer"],
            "temperature": 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, comparing choices, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])
    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
