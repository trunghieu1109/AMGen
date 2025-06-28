async def forward_168(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Extract and characterize defining features of original and variant decay

    # Sub-task 1: Extract and characterize original decay 2A -> 2B + 2E + 2V
    cot_instruction_1 = (
        "Sub-task 1: Extract and characterize the defining features of the original nuclear decay process 2A -> 2B + 2E + 2V, "
        "including the nature of particles involved (heavy nucleons A and B, lighter particles E and V), the continuous energy spectrum of the outgoing E particles, and the endpoint energy Q of this spectrum."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing original decay, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Extract and characterize variant decay 2A -> 2B + 2E + M
    cot_instruction_2 = (
        "Sub-task 2: Extract and characterize the defining features of the variant decay process where one exotic, massless particle M replaces the 2V particles, "
        "focusing on how the particle content and conservation laws might differ from the original decay."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, analyzing variant decay, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 1: Analyze impact on energy distribution and kinematics

    # Sub-task 3: Analyze effect of replacing 2V with massless M on energy distribution and kinematics
    cot_reflect_instruction_3 = (
        "Sub-task 3: Analyze how replacing the two V particles with a single massless exotic particle M affects the energy distribution and kinematics of the outgoing E particles, "
        "considering conservation of energy and momentum and the mass differences of emitted particles."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking2, answer2]
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, analyzing kinematics impact, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3, answer3],
                                               "Critically evaluate the analysis of energy distribution and kinematics impact, checking for correctness and completeness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining kinematics analysis, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Assess impact on shape and endpoint of E energy spectrum
    cot_reflect_instruction_4 = (
        "Sub-task 4: Assess how the change in emitted particles (from 2V to M) impacts the shape (continuous vs discrete) and endpoint (maximum energy Q) of the total energy spectrum of the outgoing E particles, "
        "based on the kinematic constraints identified in Sub-task 3."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking3, answer3]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, assessing spectrum shape and endpoint, thinking: {thinking4.content}; answer: {answer4.content}")

    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                               "Critically evaluate the assessment of spectrum shape and endpoint, checking for correctness and completeness.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining spectrum assessment, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Stage 2: Classify and compare with multiple-choice options

    # Sub-task 5: Classify energy spectrum and endpoint change
    cot_instruction_5 = (
        "Sub-task 5: Classify the resulting energy spectrum of the outgoing E particles in the variant decay as continuous or discrete, "
        "and determine whether the endpoint energy increases, decreases, or remains the same compared to the original decay, based on the analysis from Sub-task 4."
    )
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4, answer4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, classifying spectrum and endpoint, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Compare classification with multiple-choice options
    debate_instruction_6 = (
        "Sub-task 6: Compare the classification and endpoint energy results with the provided multiple-choice options to identify the correct description of the energy spectrum change in the variant decay. "
        "Use reasoning to select the best matching choice among the four options."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]

    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing with choices, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)

    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await final_decision_agent_6([taskInfo] + all_thinking6[-1] + all_answer6[-1],
                                                     "Sub-task 6: Make final decision on the correct multiple-choice answer.",
                                                     is_sub_task=True)
    agents.append(f"Final Decision agent, selecting final answer, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer
