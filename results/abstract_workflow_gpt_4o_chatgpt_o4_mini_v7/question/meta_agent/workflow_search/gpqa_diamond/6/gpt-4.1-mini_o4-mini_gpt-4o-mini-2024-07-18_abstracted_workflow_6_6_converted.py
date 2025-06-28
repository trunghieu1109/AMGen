async def forward_6(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze problem context and extract constants

    # Sub-task 1: Analyze the problem context and identify physical process and known parameters
    cot_instruction_1 = (
        "Sub-task 1: Analyze the problem context to identify the physical process (gamma-gamma annihilation into electron-positron pairs), "
        "the known parameters such as average CMB photon energy = 10^-3 eV, and the question about the gamma-ray energy threshold limiting their lifetime in the universe."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing problem context, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Sub-task 2: Extract and clarify relevant physical constants and particle properties
    cot_instruction_2 = (
        "Sub-task 2: Extract and clarify relevant physical constants and particle properties needed for the calculation, "
        "such as electron rest mass energy (m_e c^2 = 0.511 MeV) and the threshold condition for pair production."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, extracting physical constants, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 1: Formulate threshold condition and calculate minimum gamma-ray energy

    # Sub-task 3: Formulate threshold energy condition for gamma-ray photon
    cot_instruction_3 = (
        "Sub-task 3: Formulate the threshold energy condition for the gamma-ray photon to produce an electron-positron pair "
        "when colliding with a CMB photon of average energy 10^-3 eV, using the invariant energy relation for pair production."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, formulating threshold condition, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Calculate minimum gamma-ray photon energy from threshold condition
    cot_instruction_4 = (
        "Sub-task 4: Calculate the minimum gamma-ray photon energy from the threshold condition derived in Sub-task 3, "
        "substituting known values for electron mass and CMB photon energy."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking3, answer3], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating minimum gamma-ray energy, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    # Stage 2: Compare calculated threshold with choices and conclude

    # Sub-task 5: Compare calculated gamma-ray energy threshold with multiple-choice options
    debate_instruction_5 = (
        "Sub-task 5: Compare the calculated gamma-ray energy threshold with the provided multiple-choice options: "
        "9.5*10^4 GeV, 1.8*10^5 GeV, 2.6*10^5 GeV, 3.9*10^5 GeV, to identify the closest matching value."
    )
    debate_agents_5 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]

    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            input_infos_5 = [taskInfo, thinking4, answer4]
            if r > 0:
                input_infos_5 += all_thinking5[r-1] + all_answer5[r-1]
            thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing threshold with choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)

    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo] + all_thinking5[-1] + all_answer5[-1],
        "Sub-task 5: Make final decision on the closest matching gamma-ray energy threshold.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, deciding closest threshold, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Subtask 5 answer: ", sub_tasks[-1])

    # Sub-task 6: Conclude and state the gamma-ray energy above which lifetime is limited
    cot_instruction_6 = (
        "Sub-task 6: Conclude and state the gamma-ray energy above which their lifetime in the universe is limited by the gamma-gamma to electron-positron pair production process, "
        "based on the comparison in Sub-task 5."
    )
    cot_agent_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent_6([taskInfo, thinking5, answer5], cot_instruction_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_6.id}, concluding final gamma-ray energy threshold, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Subtask 6 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
