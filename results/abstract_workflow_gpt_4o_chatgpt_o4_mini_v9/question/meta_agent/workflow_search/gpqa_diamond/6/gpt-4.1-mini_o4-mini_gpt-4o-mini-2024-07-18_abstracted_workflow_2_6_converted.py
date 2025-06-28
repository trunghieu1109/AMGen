async def forward_6(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and classify elements
    # Sub-task 1: Analyze the physical process and identify key physical quantities
    cot_instruction_1 = (
        "Sub-task 1: Analyze the physical process of high-energy gamma-ray photon annihilation with a CMB photon "
        "into an electron-positron pair (γγ → e⁺e⁻). Identify key physical quantities such as photon energies, "
        "electron rest mass energy, and threshold conditions for pair production."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing physical process, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Extract and classify known parameters from the query
    cot_instruction_2 = (
        "Sub-task 2: Extract and classify known parameters from the query, including the average CMB photon energy (10^-3 eV) "
        "and the electron rest mass energy (0.511 MeV). Understand their roles in the pair production threshold calculation."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, extracting known parameters, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Derive threshold condition, calculate numerical value, and compare choices
    # Sub-task 3: Derive threshold energy condition for gamma-ray photon
    cot_instruction_3 = (
        "Sub-task 3: Derive the threshold energy condition for the high-energy gamma-ray photon to produce an electron-positron pair "
        "when interacting with a CMB photon of given energy, applying energy and momentum conservation and invariant mass threshold."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, deriving threshold energy condition, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Calculate numerical value of threshold gamma-ray photon energy
    cot_instruction_4 = (
        "Sub-task 4: Calculate the numerical value of the threshold gamma-ray photon energy using the derived formula "
        "and the given average CMB photon energy (10^-3 eV). Convert all units consistently to eV or GeV as needed."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating numerical threshold energy, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Sub-task 5: Compare calculated threshold energy with provided choices to identify correct option
    debate_instruction_5 = (
        "Sub-task 5: Compare the calculated threshold gamma-ray energy with the provided multiple-choice options "
        "(9.5*10^4 GeV, 1.8*10^5 GeV, 2.6*10^5 GeV, 3.9*10^5 GeV) and identify which corresponds to the energy above which gamma-rays have their lifetimes limited by this annihilation process."
    )
    debate_roles = ["Proposer", "Skeptic"]
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking5[r-1] + all_answer5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing threshold energy with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5([taskInfo] + all_thinking5[-1] + all_answer5[-1], "Sub-task 5: Make final decision on the correct threshold gamma-ray energy choice.", is_sub_task=True)
    agents.append(f"Final Decision agent, making final decision on threshold energy, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
