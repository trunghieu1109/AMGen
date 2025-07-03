async def forward_193(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Enumerate all possible spin configurations (S1, S2, S3), where each spin can be +1 or -1, to define the state space of the system." 
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
        agents.append(f"CoT-SC agent {results1['cot_agent'][idx].id}, enumerating spin configurations, thinking: {results1['list_thinking'][idx]}; answer: {results1['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_instruction_2 = "Subtask 2: For each spin configuration from Subtask 1, calculate the energy E using the formula E = -J (S1S2 + S1S3 + S2S3)."
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
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, calculating energies, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction_3 = "Subtask 3: Compute the Boltzmann factor e^(-βE) for each energy value obtained in Subtask 2, where β = 1/(kT)."
    results3 = await self.sc_cot(
        subtask_id="subtask_3",
        cot_sc_instruction=cot_instruction_3,
        input_list=[taskInfo, results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 2", "answer of subtask 2"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results3['cot_agent'][idx].id}, computing Boltzmann factors, thinking: {results3['list_thinking'][idx]}; answer: {results3['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = "Subtask 4: Sum all Boltzmann factors from Subtask 3 to obtain the partition function Z of the system."
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", results3['thinking'], results3['answer']],
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc={
            "instruction": "Subtask 4: Make final decision on the computed partition function Z.",
            "output": ["thinking", "answer"],
            "temperature": 0.0
        },
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, summing Boltzmann factors, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, calculating partition function Z, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    debate_instruction_5 = "Subtask 5: Compare the computed partition function Z with the given multiple-choice options and select the correct choice (A, B, C, or D)."
    debate_desc_5 = {
        "instruction": debate_instruction_5,
        "context": ["user query", results4['thinking'], results4['answer']],
        "input": [taskInfo, results4['thinking'], results4['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_5 = {
        "instruction": "Subtask 5: Make final decision on the correct multiple-choice answer.",
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
            agents.append(f"Debate agent {agent.id}, round {round}, comparing partition function with options, thinking: {results5['list_thinking'][round][idx].content}; answer: {results5['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {results5['thinking'].content}; answer: {results5['answer'].content}")
    sub_tasks.append(f"Subtask 5 output: thinking - {results5['thinking'].content}; answer - {results5['answer'].content}")
    logs.append(results5['subtask_desc'])

    final_answer = await self.make_final_answer(results5['thinking'], results5['answer'], sub_tasks, agents)
    return final_answer, logs
