async def forward_11(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []
    
    # Stage 0: Identify given quantities and physical principles
    # Sub-task 1: Identify and list all given physical quantities and constants relevant to the decay
    cot_instruction_1 = (
        "Sub-task 1: Identify and list all given physical quantities and constants relevant to the decay process Pi(+) → mu(+) + nu, "
        "including rest masses of Pi(+) and mu(+), and the condition that Pi(+) is stationary."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, identifying given quantities, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])
    
    # Sub-task 2: Understand and state conservation of energy and momentum in decay of stationary Pi(+)
    cot_instruction_2 = (
        "Sub-task 2: State the physical principle that total energy and momentum are conserved in the decay of a stationary Pi(+). "
        "Explain that initial momentum is zero and final momenta of mu(+) and neutrino are equal and opposite."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, stating conservation laws, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])
    
    # Stage 1: Calculate total energy and express conservation equations
    # Sub-task 3: Calculate total energy of stationary Pi(+) using rest mass energy
    cot_instruction_3 = (
        "Sub-task 3: Calculate the total energy of the stationary Pi(+) particle using its rest mass energy (E = m*c^2). "
        "Use rest mass 139.6 MeV."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating Pi(+) total energy, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])
    
    # Sub-task 4: Express conservation of energy equation
    cot_instruction_4 = (
        "Sub-task 4: Express the conservation of energy equation for the decay: total energy of Pi(+) equals the sum of total energies of mu(+) and neutrino. "
        "Use outputs from Sub-task 2 and 3."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking2, answer2, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, expressing energy conservation, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])
    
    # Sub-task 5: Express conservation of momentum equation
    cot_instruction_5 = (
        "Sub-task 5: Express the conservation of momentum equation for the decay, noting initial momentum is zero, so momenta of mu(+) and neutrino are equal and opposite."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking2, answer2], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, expressing momentum conservation, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])
    
    # Stage 2: Derive energies and kinetic energies
    # Sub-task 6: Derive energy of mu(+) using conservation laws and negligible neutrino mass
    cot_instruction_6 = (
        "Sub-task 6: Derive the energy of the mu(+) particle using conservation of energy and momentum, incorporating known rest masses and negligible neutrino mass."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking4, answer4, thinking5, answer5, thinking1, answer1], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, deriving mu(+) energy, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])
    
    # Sub-task 7: Calculate kinetic energy of mu(+) by subtracting rest mass energy
    cot_instruction_7 = (
        "Sub-task 7: Calculate the kinetic energy (KE) of the mu(+) particle by subtracting its rest mass energy from its total energy."
    )
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6, answer6, thinking1, answer1], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, calculating mu(+) KE, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    print("Step 7: ", sub_tasks[-1])
    
    # Sub-task 8: Calculate kinetic energy of neutrino from conservation laws and negligible rest mass
    cot_instruction_8 = (
        "Sub-task 8: Calculate the kinetic energy (KE) of the neutrino by subtracting its rest mass energy (approximately zero) from its total energy derived from conservation laws."
    )
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking6, answer6, thinking7, answer7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, calculating neutrino KE, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    print("Step 8: ", sub_tasks[-1])
    
    # Stage 3: Compare calculated kinetic energies with given choices
    debate_instruction_9 = (
        "Sub-task 9: Compare the calculated kinetic energies of mu(+) and neutrino with the given choices to identify the correct option. "
        "Use outputs from Sub-task 7 and 8 and the choices in taskInfo."
    )
    debate_agents_9 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_9 = self.max_round
    all_thinking9 = [[] for _ in range(N_max_9)]
    all_answer9 = [[] for _ in range(N_max_9)]
    
    for r in range(N_max_9):
        for i, agent in enumerate(debate_agents_9):
            if r == 0:
                input_infos_9 = [taskInfo, thinking7, answer7, thinking8, answer8]
            else:
                input_infos_9 = [taskInfo, thinking7, answer7, thinking8, answer8] + all_thinking9[r-1] + all_answer9[r-1]
            thinking9, answer9 = await agent(input_infos_9, debate_instruction_9, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing kinetic energies with choices, thinking: {thinking9.content}; answer: {answer9.content}")
            all_thinking9[r].append(thinking9)
            all_answer9[r].append(answer9)
    
    final_decision_agent_9 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking9, answer9 = await final_decision_agent_9([taskInfo] + all_thinking9[-1] + all_answer9[-1], "Sub-task 9: Make final decision on the correct choice of kinetic energies.", is_sub_task=True)
    agents.append(f"Final Decision agent on kinetic energies, thinking: {thinking9.content}; answer: {answer9.content}")
    sub_tasks.append(f"Sub-task 9 output: thinking - {thinking9.content}; answer - {answer9.content}")
    print("Step 9: ", sub_tasks[-1])
    
    final_answer = await self.make_final_answer(thinking9, answer9, sub_tasks, agents)
    return final_answer
