async def forward_191(self, taskInfo):
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = "Subtask 1: Extract and clearly define the physical setup and parameters: radius R of the spherical conductor, radius r of the cavity, displacement s between centers, charge +q inside the cavity, and positions of point P with distances L (from conductor center) and l (from cavity center), including the angle θ between vectors l and s." 
    results1 = await self.cot(
        subtask_id="subtask_1",
        cot_instruction=cot_instruction_1,
        input_list=[taskInfo],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context="user input"
    )
    agents.append(f"CoT agent {results1['cot_agent'].id}, extracting physical parameters, thinking: {results1['thinking'].content}; answer: {results1['answer'].content}")
    sub_tasks.append(f"Subtask 1 output: thinking - {results1['thinking'].content}; answer - {results1['answer'].content}")
    logs.append(results1['subtask_desc'])

    cot_sc_instruction_2 = "Subtask 2: Analyze the electrostatic properties of the uncharged spherical conductor with a cavity containing charge +q, focusing on the induced charges on the conductor's surface and the resulting external electric field characteristics, based on outputs from Subtask 1." 
    results2 = await self.sc_cot(
        subtask_id="subtask_2",
        cot_sc_instruction=cot_sc_instruction_2,
        input_list=[taskInfo, results1['thinking'], results1['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.5,
        context=["user query", "thinking of subtask 1", "answer of subtask 1"],
        n_repeat=self.max_sc
    )
    for idx in range(self.max_sc):
        agents.append(f"CoT-SC agent {results2['cot_agent'][idx].id}, analyzing induced charges and external field, thinking: {results2['list_thinking'][idx]}; answer: {results2['list_answer'][idx]}")
    sub_tasks.append(f"Subtask 2 output: thinking - {results2['thinking'].content}; answer - {results2['answer'].content}")
    logs.append(results2['subtask_desc'])

    cot_instruction_3 = "Subtask 3: Determine the effective charge distribution as seen from outside the conductor and derive the expression for the electric field magnitude at point P outside the conductor, considering the shielding effect and the geometry of the problem, based on outputs from Subtask 2." 
    results3 = await self.cot(
        subtask_id="subtask_3",
        cot_instruction=cot_instruction_3,
        input_list=[taskInfo, results2['thinking'], results2['answer']],
        output_fields=["thinking", "answer"],
        temperature=0.0,
        context=["user query", "thinking of subtask 2", "answer of subtask 2"]
    )
    agents.append(f"CoT agent {results3['cot_agent'].id}, deriving effective charge distribution and E field, thinking: {results3['thinking'].content}; answer: {results3['answer'].content}")
    sub_tasks.append(f"Subtask 3 output: thinking - {results3['thinking'].content}; answer - {results3['answer'].content}")
    logs.append(results3['subtask_desc'])

    debate_instruction_4 = "Subtask 4: Evaluate the given multiple-choice options for the electric field magnitude at point P using the derived expression from Subtask 3 and select the correct choice (A, B, C, or D)." 
    final_decision_instruction_4 = "Subtask 4: Make final decision on the correct multiple-choice option for the electric field magnitude at point P." 
    debate_desc_4 = {
        "instruction": debate_instruction_4,
        "context": ["user query", results3['thinking'], results3['answer']],
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "output": ["thinking", "answer"],
        "temperature": 0.5
    }
    final_decision_desc_4 = {
        "instruction": final_decision_instruction_4,
        "output": ["thinking", "answer"],
        "temperature": 0.0
    }
    results4 = await self.debate(
        subtask_id="subtask_4",
        debate_desc=debate_desc_4,
        final_decision_desc=final_decision_desc_4,
        n_repeat=self.max_round
    )
    for round in range(self.max_round):
        for idx, agent in enumerate(results4['debate_agent']):
            agents.append(f"Debate agent {agent.id}, round {round}, evaluating options, thinking: {results4['list_thinking'][round][idx].content}; answer: {results4['list_answer'][round][idx].content}")
    agents.append(f"Final Decision agent, selecting correct choice, thinking: {results4['thinking'].content}; answer: {results4['answer'].content}")
    sub_tasks.append(f"Subtask 4 output: thinking - {results4['thinking'].content}; answer - {results4['answer'].content}")
    logs.append(results4['subtask_desc'])

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'], sub_tasks, agents)
    return final_answer, logs
