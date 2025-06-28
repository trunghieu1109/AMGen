async def forward_188(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []

    # Stage 1: Analyze and define concepts and particles
    cot_instruction_1 = (
        "Sub-task 1: Analyze and define the concept of spontaneously-broken symmetry in physics, "
        "including what it means for a particle or quasiparticle to be associated with such a symmetry breaking."
    )
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, defining spontaneously-broken symmetry, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    print("Subtask 1 answer: ", sub_tasks[-1])

    cot_instruction_2 = (
        "Sub-task 2: Identify and describe the nature and physical origin of each given particle/effective particle: Magnon, Skyrmion, Pion, and Phonon, "
        "focusing on their fundamental properties and contexts in which they arise."
    )
    cot_agent_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await cot_agent_2([taskInfo], cot_instruction_2, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_2.id}, describing particles, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    print("Subtask 2 answer: ", sub_tasks[-1])

    # Stage 2: Determine association with spontaneously-broken symmetry and identify which is not associated
    cot_sc_instruction_3 = (
        "Sub-task 3: For each particle/effective particle (Magnon, Skyrmion, Pion, Phonon), determine whether it is associated with a spontaneously-broken symmetry by comparing their physical origin and characteristics "
        "against the definition and criteria established in Sub-task 1 and the descriptions from Sub-task 2."
    )
    N = self.max_sc
    cot_agents_3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_answers_3 = []
    thinkingmapping_3 = {}
    answermapping_3 = {}
    for i in range(N):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1, thinking2, answer2], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, determining association with spontaneously-broken symmetry, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3.content)
        thinkingmapping_3[answer3.content] = thinking3
        answermapping_3[answer3.content] = answer3
    counter_3 = Counter(possible_answers_3)
    most_common_answer_3 = counter_3.most_common(1)[0][0]
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinkingmapping_3[most_common_answer_3].content}; answer - {most_common_answer_3}")
    print("Subtask 3 answer: ", sub_tasks[-1])

    cot_reflect_instruction_4 = (
        "Sub-task 4: Based on the analysis in Sub-task 3, identify which of the given particles is NOT associated with a spontaneously-broken symmetry, "
        "thereby answering the original question."
    )
    cot_agent_4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_4 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_4 = self.max_round
    cot_inputs_4 = [taskInfo, thinkingmapping_3[most_common_answer_3], answermapping_3[most_common_answer_3]]
    thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_4.id}, identifying particle not associated with spontaneously-broken symmetry, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(N_max_4):
        feedback, correct = await critic_agent_4([taskInfo, thinking4, answer4],
                                               "Please review the identification of the particle not associated with spontaneously-broken symmetry and provide its limitations.",
                                               i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_4.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_4.extend([thinking4, answer4, feedback])
        thinking4, answer4 = await cot_agent_4(cot_inputs_4, cot_reflect_instruction_4, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_4.id}, refining identification, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    print("Subtask 4 answer: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer
