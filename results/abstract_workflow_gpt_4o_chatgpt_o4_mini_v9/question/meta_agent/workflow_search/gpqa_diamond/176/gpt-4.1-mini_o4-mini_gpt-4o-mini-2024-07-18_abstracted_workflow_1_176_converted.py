async def forward_176(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Analyze given problem statement to identify known parameters and physical conditions
    cot_instruction_0_1 = (
        "Sub-task 1: Analyze the problem statement to identify all known parameters and physical conditions: "
        "radius ratio (1.5), mass ratio (1.5), equal peak wavelengths (implying equal temperature), radial velocities (0 km/s and 700 km/s), "
        "and the assumption that stars radiate as black bodies."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, analyzing problem parameters, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 0.1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Classify the physical quantities involved (radius, mass, temperature, luminosity, radial velocity) "
        "and their relationships relevant to black body radiation and stellar luminosity calculations, based on Sub-task 1 output."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, classifying physical quantities, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 0.2: ", sub_tasks[-1])

    # Stage 1: Derive temperature and luminosity expressions
    cot_instruction_1_3 = (
        "Sub-task 3: Derive the temperature of both stars from the given information that their peak wavelengths are the same, "
        "using Wien's displacement law, confirming that both stars have the same temperature."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, deriving temperature, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Step 1.3: ", sub_tasks[-1])

    cot_instruction_1_4 = (
        "Sub-task 4: Express the luminosity of each star in terms of radius and temperature using the Stefan-Boltzmann law, "
        "considering the stars as black bodies."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3, answer_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, expressing luminosity, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Step 1.4: ", sub_tasks[-1])

    # Stage 2: Calculate luminosity ratio and evaluate velocity effects
    cot_instruction_2_5 = (
        "Sub-task 5: Calculate the luminosity ratio L1/L2 using the radius ratio (1.5) and the fact that temperatures are equal, "
        "applying the Stefan-Boltzmann law (L = 4πR^2σT^4)."
    )
    cot_agent_2_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2_5, answer_2_5 = await cot_agent_2_5([taskInfo, thinking_1_4, answer_1_4], cot_instruction_2_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_5.id}, calculating luminosity ratio, thinking: {thinking_2_5.content}; answer: {answer_2_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_2_5.content}; answer - {answer_2_5.content}")
    print("Step 2.5: ", sub_tasks[-1])

    cot_instruction_2_6 = (
        "Sub-task 6: Evaluate if the radial velocities (0 and 700 km/s) affect the observed luminosities or temperatures, "
        "considering Doppler shifts and relativistic effects, and determine if any correction is needed for luminosity comparison."
    )
    cot_agent_2_6 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_2_6, answer_2_6 = await cot_agent_2_6([taskInfo, thinking_0_1, answer_0_1, thinking_1_3, answer_1_3], cot_instruction_2_6, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2_6.id}, evaluating velocity effects, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking_2_6.content}; answer - {answer_2_6.content}")
    print("Step 2.6: ", sub_tasks[-1])

    # Stage 3: Integrate results and select closest luminosity ratio
    debate_instruction_3_7 = (
        "Sub-task 7: Integrate the results from luminosity ratio calculation and velocity effect evaluation to finalize the factor by which Star_1's luminosity is greater than Star_2's luminosity."
    )
    debate_agents_3_7 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3_7 = self.max_round
    all_thinking_3_7 = [[] for _ in range(N_max_3_7)]
    all_answer_3_7 = [[] for _ in range(N_max_3_7)]

    for r in range(N_max_3_7):
        for i, agent in enumerate(debate_agents_3_7):
            input_infos_3_7 = [taskInfo, thinking_2_5, answer_2_5, thinking_2_6, answer_2_6]
            if r > 0:
                input_infos_3_7 += all_thinking_3_7[r-1] + all_answer_3_7[r-1]
            thinking_3_7, answer_3_7 = await agent(input_infos_3_7, debate_instruction_3_7, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, integrating luminosity ratio and velocity effects, thinking: {thinking_3_7.content}; answer: {answer_3_7.content}")
            all_thinking_3_7[r].append(thinking_3_7)
            all_answer_3_7[r].append(answer_3_7)

    final_decision_agent_3_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_3_8, answer_3_8 = await final_decision_agent_3_8(
        [taskInfo] + all_thinking_3_7[-1] + all_answer_3_7[-1],
        "Sub-task 8: Compare the computed luminosity ratio with the provided multiple-choice options (~2.23, ~2.25, ~2.32, ~2.35) and select the closest matching value.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent, selecting closest luminosity ratio, thinking: {thinking_3_8.content}; answer: {answer_3_8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking_3_8.content}; answer - {answer_3_8.content}")
    print("Step 3.8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_3_8, answer_3_8, sub_tasks, agents)
    return final_answer
