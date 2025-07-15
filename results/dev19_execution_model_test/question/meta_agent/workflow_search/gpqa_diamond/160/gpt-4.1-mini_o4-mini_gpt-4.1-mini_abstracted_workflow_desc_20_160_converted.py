async def forward_160(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0_1 = (
        "Sub-task 0_1: Extract and summarize all given information from the query, including system parameters, "
        "vacuum conditions, initial mean free path λ1, and the observed change to λ2 upon electron beam initiation."
    )
    cot_sc_agents_0_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                      model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_0_1 = []
    possible_thinkings_0_1 = []
    subtask_desc_0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_sc_instruction_0_1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking0_1, answer0_1 = await cot_sc_agents_0_1[i]([taskInfo], cot_sc_instruction_0_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_0_1[i].id}, extracting and summarizing given info, thinking: {thinking0_1.content}; answer: {answer0_1.content}")
        possible_answers_0_1.append(answer0_1)
        possible_thinkings_0_1.append(thinking0_1)
    final_decision_agent_0_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                            model=self.node_model, temperature=0.0)
    thinking0_1, answer0_1 = await final_decision_agent_0_1(
        [taskInfo] + possible_thinkings_0_1 + possible_answers_0_1,
        "Sub-task 0_1: Synthesize and choose the most consistent summary of given information.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 0_1 output: thinking - {thinking0_1.content}; answer - {answer0_1.content}")
    subtask_desc_0_1['response'] = {"thinking": thinking0_1, "answer": answer0_1}
    logs.append(subtask_desc_0_1)
    print("Step 0_1: ", sub_tasks[-1])

    debate_instruction_1_1 = (
        "Sub-task 1_1: Analyze and integrate the physical relationships between gas molecule mean free path λ1 "
        "and the effect of electron beam-induced scattering leading to λ2, considering vacuum physics, electron scattering theory, and kinetic gas theory. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", 
                                       model=self.node_model, role=role, temperature=0.5) 
                        for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking0_1.content, answer0_1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking1_1, answer1_1 = await agent(
                    [taskInfo, thinking0_1, answer0_1], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking0_1, answer0_1] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking1_1, answer1_1 = await agent(
                    input_infos_1_1, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing physical relationships, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
            all_thinking_1_1[r].append(thinking1_1)
            all_answer_1_1[r].append(answer1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                            model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking0_1, answer0_1] + all_thinking_1_1[-1] + all_answer_1_1[-1],
        "Sub-task 1_1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1_1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1_1: ", sub_tasks[-1])

    cot_sc_instruction_2_1 = (
        "Sub-task 2_1: Assess the impact of electron beam presence on the mean free path, interpret the significance of the factor 1.22, "
        "and determine the correct inequality or equality relationship between λ1 and λ2 based on the physics involved."
    )
    cot_sc_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                      model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_answers_2_1 = []
    possible_thinkings_2_1 = []
    subtask_desc_2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_sc_instruction_2_1,
        "context": ["user query", thinking0_1.content, answer0_1.content, thinking1_1.content, answer1_1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(self.max_sc):
        thinking2_1, answer2_1 = await cot_sc_agents_2_1[i](
            [taskInfo, thinking0_1, answer0_1, thinking1_1, answer1_1], cot_sc_instruction_2_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents_2_1[i].id}, assessing electron beam impact and factor 1.22, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
        possible_answers_2_1.append(answer2_1)
        possible_thinkings_2_1.append(thinking2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", 
                                            model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1(
        [taskInfo, thinking0_1, answer0_1, thinking1_1, answer1_1] + possible_thinkings_2_1 + possible_answers_2_1,
        "Sub-task 2_1: Synthesize and choose the most consistent assessment of electron beam impact and factor 1.22.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2_1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2_1: ", sub_tasks[-1])

    reflect_inst_2_2 = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction_2_2 = (
        "Sub-task 2_2: Derive the final conclusion about λ2 relative to λ1 and select the correct choice from the given options, "
        "justifying the conclusion based on the previous analysis. " + reflect_inst_2_2
    )
    cot_agent_2_2 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", 
                                  model=self.node_model, temperature=0.0)
    critic_agent_2_2 = LLMAgentBase(["feedback", "correct"], "Critic Agent", 
                                    model=self.node_model, temperature=0.0)
    N_max_2_2 = self.max_round
    cot_inputs_2_2 = [taskInfo, thinking0_1, answer0_1, thinking1_1, answer1_1, thinking2_1, answer2_1]
    subtask_desc_2_2 = {
        "subtask_id": "subtask_2_2",
        "instruction": cot_reflect_instruction_2_2,
        "context": ["user query", thinking0_1.content, answer0_1.content, thinking1_1.content, answer1_1.content, thinking2_1.content, answer2_1.content],
        "agent_collaboration": "Reflexion"
    }
    thinking2_2, answer2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, deriving final conclusion, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    critic_inst_2_2 = (
        "Please review the answer above and criticize on where might be wrong. "
        "If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    )
    for i in range(N_max_2_2):
        feedback, correct = await critic_agent_2_2(
            [taskInfo, thinking2_2, answer2_2],
            "Please review and provide the limitations of provided solutions" + critic_inst_2_2,
            i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_2_2.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_2_2.extend([thinking2_2, answer2_2, feedback])
        thinking2_2, answer2_2 = await cot_agent_2_2(cot_inputs_2_2, cot_reflect_instruction_2_2, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_2_2.id}, refining final conclusion, thinking: {thinking2_2.content}; answer: {answer2_2.content}")
    sub_tasks.append(f"Sub-task 2_2 output: thinking - {thinking2_2.content}; answer - {answer2_2.content}")
    subtask_desc_2_2['response'] = {"thinking": thinking2_2, "answer": answer2_2}
    logs.append(subtask_desc_2_2)
    print("Step 2_2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_2, answer2_2, sub_tasks, agents)
    return final_answer, logs
