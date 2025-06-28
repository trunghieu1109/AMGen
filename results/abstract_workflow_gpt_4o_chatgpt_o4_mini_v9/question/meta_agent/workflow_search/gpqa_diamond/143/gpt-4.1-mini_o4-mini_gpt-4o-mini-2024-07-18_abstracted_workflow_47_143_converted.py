async def forward_143(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Extract and convert parameters
    # Sub-task 1: Identify and extract all given physical parameters relevant to the meson resonance X
    cot_instruction_1 = (
        "Sub-task 1: Identify and extract all given physical parameters relevant to the meson resonance X, "
        "including production energy (E_X = 8 GeV), mass (m_X = 1.2 GeV), and width (\u0393_X = 320 MeV). "
        "This sets the foundation for subsequent calculations by clearly defining the input values."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, extracting parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Convert all given parameters into consistent units (convert width from MeV to GeV)
    cot_instruction_2 = (
        "Sub-task 2: Convert all given parameters into consistent units suitable for calculation, specifically converting the width \u0393_X from MeV to GeV to match other energy units, "
        "and prepare any other unit conversions needed for later computations."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, converting units, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Calculate lifetime, gamma factor, and mean decay distance
    # Sub-task 3: Calculate lifetime (tau) using tau = hbar / Gamma
    cot_instruction_3 = (
        "Sub-task 3: Calculate the lifetime (\u03C4) of the meson resonance X using the relation \u03C4 = \u210F / \u0393, "
        "where \u210F is the reduced Planck constant. Use the width \u0393 (converted to GeV) from Sub-task 2 to find the mean lifetime in seconds."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_3([taskInfo, thinking2, answer2], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, calculating lifetime, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Calculate relativistic gamma factor gamma = E_X / m_X
    cot_instruction_4 = (
        "Sub-task 4: Calculate the relativistic gamma factor (\u03B3) for the meson resonance X using the production energy E_X and mass m_X with the formula \u03B3 = E_X / m_X. "
        "This factor accounts for time dilation effects on the lifetime due to the meson's motion."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_4([taskInfo, thinking1, answer1], cot_instruction_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_4.id}, calculating gamma factor, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: Compute mean decay distance L = gamma * c * tau
    cot_instruction_5 = (
        "Sub-task 5: Compute the mean decay distance (L) of the meson resonance X by combining the lifetime \u03C4 from Sub-task 3, the gamma factor \u03B3 from Sub-task 4, and the speed of light c, "
        "using the formula L = \u03B3 * c * \u03C4. This yields the mean decay distance in meters."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking3, answer3, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, calculating mean decay distance, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare calculated mean decay distance L with provided multiple-choice options
    debate_instruction_6 = (
        "Sub-task 6: Compare the calculated mean decay distance L with the provided multiple-choice options to identify the closest matching value. "
        "Determine the correct answer choice based on the computed result."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            input_infos_6 = [taskInfo, thinking5, answer5]
            if r > 0:
                input_infos_6.extend(all_thinking6[r-1])
            thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing mean decay distance with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1], "Sub-task 6: Make final decision on the closest matching mean decay distance choice.", is_sub_task=True)
    agents.append(f"Final Decision agent on mean decay distance choice, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
