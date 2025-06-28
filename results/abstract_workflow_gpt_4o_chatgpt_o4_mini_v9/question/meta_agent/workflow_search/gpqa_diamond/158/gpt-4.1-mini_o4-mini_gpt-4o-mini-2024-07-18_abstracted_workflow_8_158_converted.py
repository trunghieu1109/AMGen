async def forward_158(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Identify redshift and intrinsic emission wavelength

    # Sub-task 2: Confirm or estimate intrinsic emission wavelength of quasar spectral feature
    cot_instruction_sub2 = (
        "Sub-task 2: Identify the intrinsic emission wavelength of the quasar spectral feature corresponding to the observed peak at 790 nm, "
        "using typical quasar emission lines such as Lyman-alpha (121.6 nm) or others in near-infrared/optical range."
    )
    cot_agent_sub2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_sub2([taskInfo], cot_instruction_sub2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_sub2.id}, estimating intrinsic emission wavelength, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Sub-task 1: Identify redshift (z) based on observed peak wavelength (790 nm) and intrinsic emission wavelength
    cot_instruction_sub1 = (
        "Sub-task 1: Calculate the redshift z of the quasar using the relation lambda_observed = lambda_emitted * (1 + z), "
        "where lambda_observed is 790 nm and lambda_emitted is the intrinsic emission wavelength estimated in Sub-task 2."
    )
    cot_agent_sub1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_sub1([taskInfo, thinking2, answer2], cot_instruction_sub1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_sub1.id}, calculating redshift z, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    # Stage 2: Calculate comoving distance and compare with choices

    # Sub-task 3: Calculate comoving distance using redshift z and Lambda-CDM cosmology
    cot_reflect_instruction_sub3 = (
        "Sub-task 3: Using the redshift z from Sub-task 1 and the Lambda-CDM cosmological parameters "
        "(H0=70 km/s/Mpc, Omega_m=0.3, Omega_Lambda=0.7, flat universe), calculate the comoving distance to the quasar by integrating the cosmological distance formula."
    )
    cot_agent_sub3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_sub3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs_sub3 = [taskInfo, thinking1, answer1, thinking2, answer2]
    thinking3, answer3 = await cot_agent_sub3(cot_inputs_sub3, cot_reflect_instruction_sub3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_sub3.id}, calculating comoving distance, thinking: {thinking3.content}; answer: {answer3.content}")

    for i in range(N_max):
        feedback, correct = await critic_agent_sub3([taskInfo, thinking3, answer3],
                                                  "Critically evaluate the comoving distance calculation for correctness and completeness.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_sub3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_sub3.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_sub3(cot_inputs_sub3, cot_reflect_instruction_sub3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_sub3.id}, refining comoving distance, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    # Sub-task 4: Compare calculated comoving distance with given choices and select closest
    debate_instruction_sub4 = (
        "Sub-task 4: Compare the calculated comoving distance from Sub-task 3 with the given multiple-choice options (6 Gpc, 7 Gpc, 8 Gpc, 9 Gpc) "
        "and select the closest value as the assumed comoving distance of the quasar from Earth at scale factor a=1."
    )
    debate_roles = ["Pro-6Gpc", "Pro-7Gpc", "Pro-8Gpc", "Pro-9Gpc"]
    debate_agents_sub4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in debate_roles]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_sub4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3, answer3], debate_instruction_sub4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_sub4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, comparing comoving distance choices, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)

    final_decision_agent_sub4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_sub4([taskInfo] + all_thinking4[-1] + all_answer4[-1],
                                                        "Sub-task 4: Make final decision on the closest comoving distance choice.",
                                                        is_sub_task=True)
    agents.append(f"Final Decision agent, selecting closest comoving distance, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
