async def forward_160(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: CoT
    cot_instruction = (
        "Sub-task 1: Explicitly define λ₁ and λ₂, distinguishing the particle and collision processes each represents, "
        "clarifying that λ₁ is the mean free path of gas molecules between gas–gas collisions, "
        "while λ₂ is the mean free path of gas molecules under electron-beam irradiation."
    )
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: SC_CoT
    cot_sc_instruction = (
        "Sub-task 2: Summarize all given physical parameters (vacuum pressure, temperature, gas density, electron accelerating voltage) "
        "needed for quantitative analysis, ensuring the gas number density n is clearly stated."
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction,
        "context": ["user query", thinking1.content, answer1.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_agents:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings.append(thinking2_i)
        possible_answers.append(answer2_i)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr2 = (
        "Sub-task 2: Given all SC outputs, find the most consistent and correct summary of physical parameters."
    )
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + possible_thinkings + possible_answers,
        synth_instr2,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final_decision_agent_2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Debate
    debate_instruction_3 = (
        "Sub-task 3: Identify and list all collision and interaction processes that affect gas molecules when the electron beam is on (e.g. gas–gas scattering, electron–gas elastic/inelastic collisions, ionization, excitation). "
        "Embed feedback by explicitly acknowledging that beam-induced processes can add additional scattering channels and alter λ₂. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking3 = [[] for _ in range(N_max_3)]
    all_answer3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instruction_3,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for agent in debate_agents_3:
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                ctx = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(ctx, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr3 = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final list of interaction processes."
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1],
        synth_instr3,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final_decision_agent_3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: SC_CoT
    cot_sc_instruction4 = (
        "Sub-task 4: Gather realistic scattering cross-section values or credible estimates for molecular gas–gas collisions and for electron–gas interactions at 1000 kV, "
        "addressing the source or justification of the 1.22 factor."
    )
    N4 = self.max_sc
    sc_agents4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N4)]
    possible_thinkings4 = []
    possible_answers4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction4,
        "context": ["user query", thinking3.content, answer3.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in sc_agents4:
        t4, a4 = await agent([taskInfo, thinking3, answer3], cot_sc_instruction4, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t4.content}; answer: {a4.content}")
        possible_thinkings4.append(t4)
        possible_answers4.append(a4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr4 = "Sub-task 4: Synthesize the most credible cross-section estimates at 1000 kV."
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking3, answer3] + possible_thinkings4 + possible_answers4,
        synth_instr4,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final_decision_agent_4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Sub-task 5: SC_CoT
    cot_sc_instruction5 = (
        "Sub-task 5: Compute λ₁ and λ₂ quantitatively using λ=1/(n·σ_total), where σ_total for λ₂ includes all beam-induced cross sections. "
        "Calculate the ratio λ₂/λ₁ numerically."
    )
    N5 = self.max_sc
    sc_agents5 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N5)]
    possible_thinkings5 = []
    possible_answers5 = []
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_sc_instruction5,
        "context": ["user query", thinking4.content, answer4.content],
        "agent_collaboration": "SC_CoT"
    }
    for agent in sc_agents5:
        t5, a5 = await agent([taskInfo, thinking4, answer4], cot_sc_instruction5, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {t5.content}; answer: {a5.content}")
        possible_thinkings5.append(t5)
        possible_answers5.append(a5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr5 = "Sub-task 5: Synthesize the numerical values for λ₁, λ₂, and their ratio."
    thinking5, answer5 = await final_decision_agent_5(
        [taskInfo, thinking4, answer4] + possible_thinkings5 + possible_answers5,
        synth_instr5,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final_decision_agent_5.id}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Sub-task 6: Debate
    debate_instruction_6 = (
        "Sub-task 6: Compare the computed ratio λ₂/λ₁ to the provided options and select the correct inequality for λ₂ relative to λ₁. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", thinking5.content, answer5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for agent in debate_agents_6:
            if r == 0:
                t6_i, a6_i = await agent([taskInfo, thinking5, answer5], debate_instruction_6, r, is_sub_task=True)
            else:
                ctx6 = [taskInfo, thinking5, answer5] + all_thinking6[r-1] + all_answer6[r-1]
                t6_i, a6_i = await agent(ctx6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {t6_i.content}; answer: {a6_i.content}")
            all_thinking6[r].append(t6_i)
            all_answer6[r].append(a6_i)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr6 = "Sub-task 6: Given all the above thinking and answers, reason over them carefully and provide a final inequality choice."
    thinking6, answer6 = await final_decision_agent_6(
        [taskInfo, thinking5, answer5] + all_thinking6[-1] + all_answer6[-1],
        synth_instr6,
        is_sub_task=True
    )
    agents.append(f"Final Decision Agent {final_decision_agent_6.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs