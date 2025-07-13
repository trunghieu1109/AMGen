async def forward_177(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []
    # Stage 1: extract field mass dimensions using SC_CoT
    cot_sc_instruction = "Sub-task 1: Extract the mass dimension of the Dirac spinor psi, its adjoint psi_bar, and the field strength F^mu nu in four dimensions with hbar=c=1."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings.append(thinking1)
        possible_answers.append(answer1)
    final_sc = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_sc([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize and choose the most consistent answer for the field dimensions.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])
    # Stage 2: determine mass dimension of kappa using Debate
    debate_instruction_2 = "Sub-task 2: Determine the mass dimension of kappa given L_int = kappa psi_bar sigma_{mu nu} psi F^mu nu using the field dimensions from subtask 1. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_2 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking2 = [[] for _ in range(self.max_round)]
    all_answer2 = [[] for _ in range(self.max_round)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":debate_instruction_2,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                inputs2 = [taskInfo, thinking1, answer1]
            else:
                inputs2 = [taskInfo, thinking1, answer1] + all_thinking2[r-1] + all_answer2[r-1]
            thinking2_i, answer2_i = await agent(inputs2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
            all_thinking2[r].append(thinking2_i)
            all_answer2[r].append(answer2_i)
    final_decision_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_2([taskInfo, thinking1, answer1] + all_thinking2[-1] + all_answer2[-1], "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])
    # Stage 3: determine renormalizability using Debate
    debate_instruction_3 = "Sub-task 3: Determine whether the interaction is renormalizable by checking if the mass dimension of kappa is >= 0 in four dimensions and match the result to one of the provided choices. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instruction_3,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                inputs3 = [taskInfo, thinking2, answer2]
            else:
                inputs3 = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
            thinking3_i, answer3_i = await agent(inputs3, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_decision_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_3([taskInfo, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3, "answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])
    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs