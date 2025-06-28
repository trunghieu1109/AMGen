async def forward_191(self, taskInfo):
    from collections import Counter
    
    print("Task Requirement: ", taskInfo)
    
    sub_tasks = []
    agents = []

    # Stage 0: Extract and define physical parameters and setup
    cot_instruction_0_1 = (
        "Sub-task 1: Extract and clearly define all given physical parameters and geometric relationships from the problem statement, "
        "including the spherical conductor radius R, cavity radius r, displacement s between centers, charge +q inside the cavity, "
        "point P location parameters L and l, and the angle theta between vectors l and s. Identify the conditions l, L > R and r < R."
    )
    cot_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_1, answer_0_1 = await cot_agent_0_1([taskInfo], cot_instruction_0_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_1.id}, extracting physical parameters, thinking: {thinking_0_1.content}; answer: {answer_0_1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking_0_1.content}; answer - {answer_0_1.content}")
    print("Step 1: ", sub_tasks[-1])

    cot_instruction_0_2 = (
        "Sub-task 2: Identify the physical setup and constraints: an uncharged spherical conductor with a cavity containing a charge +q, "
        "and the point P outside the conductor where the electric field is to be found. Clarify the significance of the conductor being uncharged "
        "and the cavity placement inside it."
    )
    cot_agent_0_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_0_2, answer_0_2 = await cot_agent_0_2([taskInfo, thinking_0_1, answer_0_1], cot_instruction_0_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0_2.id}, identifying physical setup, thinking: {thinking_0_2.content}; answer: {answer_0_2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking_0_2.content}; answer - {answer_0_2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Analyze electrostatic behavior and apply method of images
    cot_instruction_1_3 = (
        "Sub-task 3: Analyze the electrostatic behavior of the conductor with the cavity and internal charge +q, "
        "including the induced charge distribution on the conductor surface and the effect on the external electric field at point P. "
        "Use the fact that the conductor is uncharged overall but can have induced charges due to the internal charge."
    )
    cot_agent_1_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_3, answer_1_3 = await cot_agent_1_3([taskInfo, thinking_0_2, answer_0_2], cot_instruction_1_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_3.id}, analyzing electrostatic behavior, thinking: {thinking_1_3.content}; answer: {answer_1_3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking_1_3.content}; answer - {answer_1_3.content}")
    print("Step 3: ", sub_tasks[-1])

    cot_instruction_1_4 = (
        "Sub-task 4: Apply the method of images or equivalent electrostatic principles to determine the effective charge distribution outside the conductor "
        "and the resulting electric field at point P, considering the displacement s and angle theta between cavity center and point P."
    )
    cot_agent_1_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_4, answer_1_4 = await cot_agent_1_4([taskInfo, thinking_1_3, answer_1_3], cot_instruction_1_4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_4.id}, applying method of images, thinking: {thinking_1_4.content}; answer: {answer_1_4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking_1_4.content}; answer - {answer_1_4.content}")
    print("Step 4: ", sub_tasks[-1])

    cot_instruction_1_5 = (
        "Sub-task 5: Express the electric field magnitude at point P in terms of the given parameters q, l, s, theta, and constants, "
        "and simplify the expression to identify the correct functional form of E."
    )
    cot_agent_1_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking_1_5, answer_1_5 = await cot_agent_1_5([taskInfo, thinking_1_4, answer_1_4], cot_instruction_1_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1_5.id}, expressing and simplifying E, thinking: {thinking_1_5.content}; answer: {answer_1_5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking_1_5.content}; answer - {answer_1_5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Stage 2: Compare with options and conclude
    debate_instruction_2_6 = (
        "Sub-task 6: Compare the derived expression for the electric field magnitude at point P with the provided multiple-choice options, "
        "analyzing each option's dependence on l, s, theta, and L to determine which matches the physical and mathematical result."
    )
    debate_agents_2_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking_2_6 = [[] for _ in range(N_max_6)]
    all_answer_2_6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_2_6):
            input_infos_2_6 = [taskInfo, thinking_1_5, answer_1_5]
            if r > 0:
                input_infos_2_6.extend(all_thinking_2_6[r-1])
            thinking_2_6, answer_2_6 = await agent(input_infos_2_6, debate_instruction_2_6, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing options, thinking: {thinking_2_6.content}; answer: {answer_2_6.content}")
            all_thinking_2_6[r].append(thinking_2_6)
            all_answer_2_6[r].append(answer_2_6)

    final_decision_agent_2_7 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking_2_7, answer_2_7 = await final_decision_agent_2_7([taskInfo] + all_thinking_2_6[-1] + all_answer_2_6[-1], "Sub-task 7: Conclude the correct formula for the electric field magnitude at point P outside the spherical conductor with the cavity and internal charge, based on the comparison and physical reasoning.", is_sub_task=True)
    agents.append(f"Final Decision agent on concluding correct formula, thinking: {thinking_2_7.content}; answer: {answer_2_7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking_2_7.content}; answer - {answer_2_7.content}")
    print("Step 7: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking_2_7, answer_2_7, sub_tasks, agents)
    return final_answer
