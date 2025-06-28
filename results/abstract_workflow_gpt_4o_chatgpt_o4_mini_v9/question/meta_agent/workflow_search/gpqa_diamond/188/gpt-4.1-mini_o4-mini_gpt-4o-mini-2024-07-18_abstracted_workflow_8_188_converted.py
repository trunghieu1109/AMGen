async def forward_188(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Foundational Context and Background Information

    # Sub-task 1: Define and explain spontaneously-broken symmetry in physics
    cot_instruction_1 = (
        "Sub-task 1: Define and explain the concept of spontaneously-broken symmetry in physics, "
        "including what it means for a particle or excitation to be associated with such a symmetry breaking. "
        "This provides the foundational context needed to evaluate each particle's association with spontaneously-broken symmetry."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining spontaneously-broken symmetry, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Identify and describe each given particle/effective excitation
    cot_instruction_2 = (
        "Sub-task 2: Identify and describe the nature of each given particle/effective excitation (Magnon, Skyrmion, Pion, Phonon), "
        "focusing on their physical origin and typical contexts in which they appear. "
        "This step gathers necessary background information on each choice, based on the definition from Sub-task 1."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo, thinking1, answer1], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, describing particles, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Analysis and Synthesis

    # Sub-task 3: Determine association of each particle with spontaneously-broken symmetry
    cot_sc_instruction_3 = (
        "Sub-task 3: For each particle (Magnon, Skyrmion, Pion, Phonon), determine whether it is associated with a spontaneously-broken symmetry "
        "by analyzing their physical origin in relation to the concept defined in Sub-task 1. "
        "Match each particle's origin to the criteria of spontaneous symmetry breaking."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyzing particle association with spontaneous symmetry breaking, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    # Aggregate the most consistent answer
    answer_counts = Counter(possible_answers_3)
    most_common_answer = answer_counts.most_common(1)[0][0]
    thinking3_final = thinkingmapping_3[most_common_answer]
    answer3_final = answermapping_3[most_common_answer]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compare results to identify which particle is NOT associated with spontaneously-broken symmetry
    cot_reflect_instruction_4 = (
        "Sub-task 4: Compare the results from Sub-task 3 to identify which particle among Magnon, Skyrmion, Pion, and Phonon "
        "is NOT associated with a spontaneously-broken symmetry. Synthesize the analysis to answer the original question."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3_final, answer3_final]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, synthesizing final identification, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback4, correct4 = await critic_agent_4([taskInfo, thinking4, answer4],
                                                  "Please review the identification of the particle NOT associated with spontaneously-broken symmetry and provide limitations.",
                                                  i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback4.content}; answer: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining final identification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    # Final answer generation
    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
