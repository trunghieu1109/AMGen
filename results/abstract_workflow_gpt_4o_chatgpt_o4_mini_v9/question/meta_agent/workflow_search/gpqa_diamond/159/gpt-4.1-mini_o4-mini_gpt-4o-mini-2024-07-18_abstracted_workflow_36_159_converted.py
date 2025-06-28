async def forward_159(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 0: Extract and define geometric and physical parameters
    cot_instruction_0 = (
        "Sub-task 1: Extract and define the geometric and physical parameters of the aperture and incident light. "
        "Identify that the aperture is an N-sided polygon with equal apothems of length a, the light is monochromatic with wavelength λ, "
        "incident along the z-axis, and the far-field diffraction pattern produces intensity maxima and minima. Also note the condition that N tends to infinity and small angle approximation (tan θ ≈ θ) applies."
    )
    cot_agent_0 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_0([taskInfo], cot_instruction_0, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_0.id}, extracting geometric and physical parameters, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 0: ", sub_tasks[-1])

    # Stage 1: Apply transformation of polygon aperture to circular aperture
    cot_instruction_1 = (
        "Sub-task 2: Apply the transformation of the polygon aperture shape as N approaches infinity to approximate the aperture as a circular aperture with radius equal to the apothem length a, "
        "using the information from Sub-task 1."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_1 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_1 = self.max_round
    cot_inputs_1 = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_agent_1(cot_inputs_1, cot_instruction_1, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_1.id}, approximating polygon aperture as circular aperture, thinking: {thinking2.content}; answer: {answer2.content}")

    for i in range(N_max_1):
        feedback, correct = await critic_agent_1([taskInfo, thinking2, answer2],
                                                "Critically evaluate the approximation of polygon aperture to circular aperture and provide limitations.",
                                                i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_1.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_1.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_agent_1(cot_inputs_1, cot_instruction_1, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_1.id}, refining circular aperture approximation, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 3: Formulate far-field diffraction pattern for circular aperture
    cot_instruction_2 = (
        "Sub-task 3: Formulate the far-field diffraction pattern for a circular aperture illuminated by monochromatic light of wavelength λ, "
        "using the radius a as the aperture size, based on the approximation from Sub-task 2 and the physical setup from Sub-task 1."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await cot_agent_2([taskInfo, thinking1, answer1, thinking2, answer2], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, formulating diffraction pattern, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Analyze diffraction pattern to find angular positions of first two minima
    cot_instruction_3 = (
        "Sub-task 4: Analyze the diffraction pattern of the circular aperture to determine the angular positions of the first two minima in the far field, "
        "using the small angle approximation (θ ≈ tan θ) and the known formula for minima in circular aperture diffraction."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await cot_agent_3([taskInfo, thinking3, answer3], cot_instruction_3, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_3.id}, analyzing angular positions of minima, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 5: Calculate angular distance between first two minima and compare with choices
    debate_instruction_4 = (
        "Sub-task 5: Calculate the angular distance between the first two minima using the results from Sub-task 4, express it in terms of λ and a, "
        "then compare the result with the given choices to identify the correct answer."
    )
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]

    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4, answer4], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4, answer4] + all_thinking4[r-1] + all_answer4[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, calculating angular distance and comparing choices, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking4[r].append(thinking5)
            all_answer4[r].append(answer5)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1],
                                                    "Sub-task 5: Make final decision on the angular distance between the first two minima and select the correct choice.",
                                                    is_sub_task=True)
    agents.append(f"Final Decision agent, selecting correct angular distance, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking5, answer5, sub_tasks, agents)
    return final_answer
